//! CPU companion for Meteo's Apple-Silicon MLX NNUE trainer.
//!
//! Tatara remains the single source of truth for PackedSfenValue decoding,
//! HalfKA_hm2 feature indices, progress8kpabs routing, NNUE quantisation, and
//! YaneuraOu serialization.  This binary only streams fixed-size sparse
//! batches to Python/MLX and exports an MLX safetensors checkpoint.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::PathBuf;

use nnue_format::layerstack_weights::LayerStackWeights;
use nnue_format::save_yaneuraou;
use nnue_train::dataloader::{PSV_RECORD_BYTES, PsvFileLoader};
use nnue_train::init::{self, LayerStackInit, WeightShape};
use rayon::prelude::*;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors, serialize_to_file};
use shogi_features::{FeatureSet, ShogiProgressKPAbs};

const PROTOCOL_MAGIC: &[u8; 8] = b"MTMLX001";
const QUANTIZED_PROTOCOL_MAGIC: &[u8; 8] = b"MTQNT001";
const PROTOCOL_VERSION: u32 = 1;
const MAX_ACTIVE: usize = 40;
const FT_IN: usize = 73_305;
const PIECE_INPUTS: usize = 1_629;
const FT_OUT: usize = 1_024;
const L1_OUT: usize = 16;
const L2_IN: usize = 30;
const L2_OUT: usize = 64;
const NUM_BUCKETS: usize = 9;

#[cfg(not(target_endian = "little"))]
compile_error!("meteo-mlx-native safetensors protocol requires little-endian targets");

type AnyError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone)]
struct FeatureRow {
    stm: [i32; MAX_ACTIVE],
    nstm: [i32; MAX_ACTIVE],
    nnz: u8,
    bucket: u8,
    score: i16,
}

fn invalid(message: impl Into<String>) -> AnyError {
    io::Error::new(io::ErrorKind::InvalidInput, message.into()).into()
}

fn parse_options(raw: &[String]) -> Result<BTreeMap<String, String>, AnyError> {
    if !raw.len().is_multiple_of(2) {
        return Err(invalid("options must be supplied as --name value pairs"));
    }
    let mut result = BTreeMap::new();
    for pair in raw.chunks_exact(2) {
        let Some(name) = pair[0].strip_prefix("--") else {
            return Err(invalid(format!("expected --name, got {}", pair[0])));
        };
        if name.is_empty() || result.insert(name.to_string(), pair[1].clone()).is_some() {
            return Err(invalid(format!("empty or duplicate option --{name}")));
        }
    }
    Ok(result)
}

fn required(options: &BTreeMap<String, String>, name: &str) -> Result<String, AnyError> {
    options
        .get(name)
        .cloned()
        .ok_or_else(|| invalid(format!("missing --{name}")))
}

fn required_path(options: &BTreeMap<String, String>, name: &str) -> Result<PathBuf, AnyError> {
    Ok(PathBuf::from(required(options, name)?))
}

fn required_usize(options: &BTreeMap<String, String>, name: &str) -> Result<usize, AnyError> {
    let raw = required(options, name)?;
    raw.parse::<usize>()
        .map_err(|error| invalid(format!("invalid --{name}={raw}: {error}")))
}

fn required_u64(options: &BTreeMap<String, String>, name: &str) -> Result<u64, AnyError> {
    let raw = required(options, name)?;
    raw.parse::<u64>()
        .map_err(|error| invalid(format!("invalid --{name}={raw}: {error}")))
}

fn check_exact_options(
    options: &BTreeMap<String, String>,
    expected: &[&str],
) -> Result<(), AnyError> {
    for name in options.keys() {
        if !expected.contains(&name.as_str()) {
            return Err(invalid(format!("unknown option --{name}")));
        }
    }
    Ok(())
}

fn feature_row(
    psv: &shogi_format::PackedSfenValue,
    progress: ShogiProgressKPAbs,
) -> Result<FeatureRow, AnyError> {
    let board = psv.decode();
    let feature_set = FeatureSet::HalfKaHmMerged.spec();
    let mut stm = [-1_i32; MAX_ACTIVE];
    let mut nstm = [-1_i32; MAX_ACTIVE];
    let written = feature_set.extract_active_features(&board, &mut stm, &mut nstm);
    if written > MAX_ACTIVE {
        return Err(invalid(format!(
            "HalfKA_hm2 emitted {written} active features, maximum is {MAX_ACTIVE}"
        )));
    }
    if stm[..written]
        .iter()
        .chain(nstm[..written].iter())
        .any(|&index| index < 0 || index as usize >= FT_IN)
    {
        return Err(invalid("HalfKA_hm2 emitted an out-of-range feature index"));
    }
    Ok(FeatureRow {
        stm,
        nstm,
        nnz: u8::try_from(written).map_err(|_| invalid("active feature count exceeds u8"))?,
        bucket: progress.bucket_board(&board, NUM_BUCKETS),
        score: board.score,
    })
}

fn push_i32(output: &mut Vec<u8>, value: i32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_i16(output: &mut Vec<u8>, value: i16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn write_feature_batch<W: Write>(writer: &mut W, rows: &[FeatureRow]) -> io::Result<()> {
    writer.write_all(&(rows.len() as u32).to_le_bytes())?;
    let mut payload = Vec::with_capacity(rows.len() * (MAX_ACTIVE * 8 + 4));
    for row in rows {
        for &value in &row.stm {
            push_i32(&mut payload, value);
        }
    }
    for row in rows {
        for &value in &row.nstm {
            push_i32(&mut payload, value);
        }
    }
    payload.extend(rows.iter().map(|row| row.nnz));
    payload.extend(rows.iter().map(|row| row.bucket));
    for row in rows {
        push_i16(&mut payload, row.score);
    }
    writer.write_all(&payload)
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding, `values` remains borrowed for the returned
    // slice lifetime, and the byte length is checked for overflow.
    let byte_len = values
        .len()
        .checked_mul(size_of::<f32>())
        .expect("f32 initializer byte length overflow");
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len) }
}

fn initial_tensor<'a>(values: &'a [f32], shape: &[usize]) -> Result<TensorView<'a>, AnyError> {
    Ok(TensorView::new(
        Dtype::F32,
        shape.to_vec(),
        f32_bytes(values),
    )?)
}

fn initialise(options: &BTreeMap<String, String>) -> Result<(), AnyError> {
    check_exact_options(options, &["output"])?;
    let output = required_path(options, "output")?;
    if output.exists() {
        return Err(invalid(format!(
            "refusing to overwrite Tatara initializer output: {}",
            output.display()
        )));
    }
    let init_spec = LayerStackInit::default_uniform();
    let ft_real = init::sample(WeightShape::flat(FT_IN * FT_OUT, FT_IN), &init_spec.ft_w);
    // Tatara appends zero virtual factorizer rows after sampling the real FT
    // block.  Giving this block independent noise would change step-0 FT
    // variance and is therefore forbidden here.
    let ft_virtual = vec![0.0_f32; PIECE_INPUTS * FT_OUT];
    let ft_b = init::sample(WeightShape::flat(FT_OUT, FT_IN), &init_spec.ft_b);
    let l1_w = init::sample(
        WeightShape::bucketed(NUM_BUCKETS * L1_OUT * FT_OUT, NUM_BUCKETS, FT_OUT),
        &init_spec.l1_w,
    );
    let l1_b = init::sample(
        WeightShape::bucketed(NUM_BUCKETS * L1_OUT, NUM_BUCKETS, FT_OUT),
        &init_spec.l1_b,
    );
    let l1f_w = init::sample(WeightShape::flat(FT_OUT * L1_OUT, FT_OUT), &init_spec.l1f_w);
    let l1f_b = init::sample(WeightShape::flat(L1_OUT, FT_OUT), &init_spec.l1f_b);
    let l2_w = init::sample(
        WeightShape::bucketed(NUM_BUCKETS * L2_OUT * L2_IN, NUM_BUCKETS, L2_IN),
        &init_spec.l2_w,
    );
    let l2_b = init::sample(
        WeightShape::bucketed(NUM_BUCKETS * L2_OUT, NUM_BUCKETS, L2_IN),
        &init_spec.l2_b,
    );
    let l3_w = init::sample(
        WeightShape::bucketed(NUM_BUCKETS * L2_OUT, NUM_BUCKETS, L2_OUT),
        &init_spec.l3_w,
    );
    let l3_b = init::sample(WeightShape::flat(NUM_BUCKETS, L2_OUT), &init_spec.l3_b);
    let tensors = vec![
        ("ft_real", initial_tensor(&ft_real, &[FT_IN, FT_OUT])?),
        (
            "ft_virtual",
            initial_tensor(&ft_virtual, &[PIECE_INPUTS, FT_OUT])?,
        ),
        ("ft_b", initial_tensor(&ft_b, &[FT_OUT])?),
        (
            "l1_w",
            initial_tensor(&l1_w, &[NUM_BUCKETS, L1_OUT, FT_OUT])?,
        ),
        ("l1_b", initial_tensor(&l1_b, &[NUM_BUCKETS, L1_OUT])?),
        ("l1f_w", initial_tensor(&l1f_w, &[FT_OUT, L1_OUT])?),
        ("l1f_b", initial_tensor(&l1f_b, &[L1_OUT])?),
        (
            "l2_w",
            initial_tensor(&l2_w, &[NUM_BUCKETS, L2_OUT, L2_IN])?,
        ),
        ("l2_b", initial_tensor(&l2_b, &[NUM_BUCKETS, L2_OUT])?),
        ("l3_w", initial_tensor(&l3_w, &[NUM_BUCKETS, L2_OUT])?),
        ("l3_b", initial_tensor(&l3_b, &[NUM_BUCKETS])?),
    ];
    serialize_to_file(tensors, None, &output)?;
    eprintln!(
        "meteo-mlx-native wrote exact Tatara LayerStack initialization to {}",
        output.display()
    );
    Ok(())
}

fn stream(options: &BTreeMap<String, String>) -> Result<(), AnyError> {
    check_exact_options(
        options,
        &[
            "input",
            "progress",
            "start-record",
            "records",
            "batch-size",
            "threads",
        ],
    )?;
    let input = required_path(options, "input")?;
    let progress_path = required_path(options, "progress")?;
    let start_record = required_u64(options, "start-record")?;
    let records = required_u64(options, "records")?;
    let batch_size = required_usize(options, "batch-size")?;
    let threads = required_usize(options, "threads")?;
    if records == 0 || batch_size == 0 || threads == 0 || threads > 256 {
        return Err(invalid(
            "records, batch-size, and threads must be positive (threads <= 256)",
        ));
    }
    let byte_size = fs::metadata(&input)?.len();
    if !byte_size.is_multiple_of(PSV_RECORD_BYTES) {
        return Err(invalid(
            "input size is not a multiple of the 40-byte PSV record",
        ));
    }
    let end_record = start_record
        .checked_add(records)
        .ok_or_else(|| invalid("record range overflows u64"))?;
    if end_record > byte_size / PSV_RECORD_BYTES {
        return Err(invalid(format!(
            "record range [{start_record}, {end_record}) exceeds {} records",
            byte_size / PSV_RECORD_BYTES
        )));
    }
    let progress = ShogiProgressKPAbs::load_from_bin(&progress_path).map_err(invalid)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("meteo-psv-{index}"))
        .build()?;
    let mut loader = PsvFileLoader::new_range(
        &input,
        start_record * PSV_RECORD_BYTES,
        end_record * PSV_RECORD_BYTES,
    )?;
    let stdout = io::stdout();
    let mut writer = BufWriter::with_capacity(32 * 1024 * 1024, stdout.lock());
    writer.write_all(PROTOCOL_MAGIC)?;
    writer.write_all(&PROTOCOL_VERSION.to_le_bytes())?;
    writer.write_all(&(MAX_ACTIVE as u32).to_le_bytes())?;
    let mut emitted = 0_u64;
    while emitted < records {
        let count = usize::try_from((records - emitted).min(batch_size as u64))
            .map_err(|_| invalid("batch record count does not fit usize"))?;
        let mut packed = Vec::with_capacity(count);
        for _ in 0..count {
            packed.push(
                loader
                    .next_psv()?
                    .ok_or_else(|| invalid("unexpected EOF inside validated PSV range"))?,
            );
        }
        let rows: Result<Vec<FeatureRow>, AnyError> = pool.install(|| {
            packed
                .par_iter()
                .map(|psv| feature_row(psv, progress))
                .collect()
        });
        write_feature_batch(&mut writer, &rows?)?;
        emitted += count as u64;
    }
    writer.write_all(&0_u32.to_le_bytes())?;
    writer.flush()?;
    eprintln!(
        "meteo-mlx-native streamed {emitted} PSV records from {} with {threads} threads",
        input.display()
    );
    Ok(())
}

fn quant_i32(value: f32, scale: i32) -> i32 {
    (value * scale as f32).round() as i32
}

fn pairwise_crelu_to_u8(input: &[i32], output: &mut [u8]) {
    let half = input.len() / 2;
    assert_eq!(output.len(), half, "pairwise output dimension changed");
    for index in 0..half {
        let first = input[index].clamp(0, 127);
        let second = input[half + index].clamp(0, 127);
        output[index] = ((first * second) >> 7).clamp(0, 126) as u8;
    }
}

fn crelu_i32_to_u8(value: i32) -> u8 {
    (value >> 6).clamp(0, 127) as u8
}

fn l1_sqr_clipped_relu(input: &[i32], output: &mut [u8]) {
    assert_eq!(output.len(), input.len() * 2, "L2 input dimension changed");
    for (index, &value) in input.iter().enumerate() {
        let value_i64 = value as i64;
        output[index] = ((value_i64 * value_i64) >> 19).clamp(0, 127) as u8;
        output[input.len() + index] = crelu_i32_to_u8(value);
    }
}

fn affine_u8_i8(input: &[u8], weights: &[f32], bias: &[f32], output: &mut [i32]) {
    let input_size = input.len();
    assert_eq!(
        weights.len(),
        output.len() * input_size,
        "quantised affine weight dimensions changed"
    );
    assert_eq!(
        bias.len(),
        output.len(),
        "quantised affine bias dimensions changed"
    );
    for output_index in 0..output.len() {
        let mut sum = quant_i32(bias[output_index], 127 * 64);
        let row = &weights[output_index * input_size..(output_index + 1) * input_size];
        for (&input_value, &weight) in input.iter().zip(row) {
            sum += input_value as i32 * quant_i32(weight, 64);
        }
        output[output_index] = sum;
    }
}

fn quantised_forward(weights: &LayerStackWeights, row: &FeatureRow) -> i32 {
    let active = row.nnz as usize;
    let mut stm_ft = vec![0_i32; FT_OUT];
    let mut nstm_ft = vec![0_i32; FT_OUT];
    for (&stm, &nstm) in row.stm[..active].iter().zip(&row.nstm[..active]) {
        let stm_offset = stm as usize * FT_OUT;
        let nstm_offset = nstm as usize * FT_OUT;
        for output in 0..FT_OUT {
            stm_ft[output] += quant_i32(weights.ft_w[stm_offset + output], 127);
            nstm_ft[output] += quant_i32(weights.ft_w[nstm_offset + output], 127);
        }
    }
    for ((stm_value, nstm_value), &bias) in
        stm_ft.iter_mut().zip(nstm_ft.iter_mut()).zip(&weights.ft_b)
    {
        let quantised_bias = quant_i32(bias, 127);
        *stm_value += quantised_bias;
        *nstm_value += quantised_bias;
    }
    let mut transformed = vec![0_u8; FT_OUT];
    pairwise_crelu_to_u8(&stm_ft, &mut transformed[..FT_OUT / 2]);
    pairwise_crelu_to_u8(&nstm_ft, &mut transformed[FT_OUT / 2..]);

    let bucket = row.bucket as usize;
    let mut l1_total = vec![0_i32; L1_OUT];
    affine_u8_i8(
        &transformed,
        &weights.l1_w[bucket * L1_OUT * FT_OUT..(bucket + 1) * L1_OUT * FT_OUT],
        &weights.l1_b[bucket * L1_OUT..(bucket + 1) * L1_OUT],
        &mut l1_total,
    );
    let mut l2_input = vec![0_u8; L2_IN];
    l1_sqr_clipped_relu(&l1_total[..L1_OUT - 1], &mut l2_input);
    let mut l2_dense = vec![0_i32; L2_OUT];
    affine_u8_i8(
        &l2_input,
        &weights.l2_w[bucket * L2_OUT * L2_IN..(bucket + 1) * L2_OUT * L2_IN],
        &weights.l2_b[bucket * L2_OUT..(bucket + 1) * L2_OUT],
        &mut l2_dense,
    );
    let l2_activated = l2_dense
        .into_iter()
        .map(crelu_i32_to_u8)
        .collect::<Vec<_>>();
    let mut output = [0_i32; 1];
    affine_u8_i8(
        &l2_activated,
        &weights.l3_w[bucket * L2_OUT..(bucket + 1) * L2_OUT],
        &weights.l3_b[bucket..bucket + 1],
        &mut output,
    );
    output[0] + l1_total[L1_OUT - 1]
}

fn evaluate_quantised(options: &BTreeMap<String, String>) -> Result<(), AnyError> {
    check_exact_options(
        options,
        &[
            "network",
            "input",
            "progress",
            "start-record",
            "records",
            "batch-size",
            "threads",
        ],
    )?;
    let network_path = required_path(options, "network")?;
    let input = required_path(options, "input")?;
    let progress_path = required_path(options, "progress")?;
    let start_record = required_u64(options, "start-record")?;
    let records = required_u64(options, "records")?;
    let batch_size = required_usize(options, "batch-size")?;
    let threads = required_usize(options, "threads")?;
    if records == 0 || batch_size == 0 || threads == 0 || threads > 256 {
        return Err(invalid(
            "records, batch-size, and threads must be positive (threads <= 256)",
        ));
    }
    let byte_size = fs::metadata(&input)?.len();
    if !byte_size.is_multiple_of(PSV_RECORD_BYTES) {
        return Err(invalid(
            "input size is not a multiple of the 40-byte PSV record",
        ));
    }
    let end_record = start_record
        .checked_add(records)
        .ok_or_else(|| invalid("record range overflows u64"))?;
    if end_record > byte_size / PSV_RECORD_BYTES {
        return Err(invalid("quantised evaluation range exceeds the PSV file"));
    }
    let mut network_reader = BufReader::new(File::open(&network_path)?);
    let weights = LayerStackWeights::load_quantised(
        &mut network_reader,
        FeatureSet::HalfKaHmMerged.spec(),
        FT_OUT,
        L1_OUT,
        L2_OUT,
        NUM_BUCKETS,
    )?;
    let progress = ShogiProgressKPAbs::load_from_bin(&progress_path).map_err(invalid)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("meteo-quantised-{index}"))
        .build()?;
    let mut loader = PsvFileLoader::new_range(
        &input,
        start_record * PSV_RECORD_BYTES,
        end_record * PSV_RECORD_BYTES,
    )?;
    let stdout = io::stdout();
    let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, stdout.lock());
    writer.write_all(QUANTIZED_PROTOCOL_MAGIC)?;
    writer.write_all(&PROTOCOL_VERSION.to_le_bytes())?;
    let mut emitted = 0_u64;
    while emitted < records {
        let count = usize::try_from((records - emitted).min(batch_size as u64))
            .map_err(|_| invalid("quantised batch count does not fit usize"))?;
        let mut packed = Vec::with_capacity(count);
        for _ in 0..count {
            packed.push(
                loader
                    .next_psv()?
                    .ok_or_else(|| invalid("unexpected EOF inside quantised PSV range"))?,
            );
        }
        let rows: Result<Vec<FeatureRow>, AnyError> = pool.install(|| {
            packed
                .par_iter()
                .map(|psv| feature_row(psv, progress))
                .collect()
        });
        let rows = rows?;
        let raw_values = pool.install(|| {
            rows.par_iter()
                .map(|row| quantised_forward(&weights, row))
                .collect::<Vec<_>>()
        });
        writer.write_all(&(count as u32).to_le_bytes())?;
        for value in raw_values {
            writer.write_all(&value.to_le_bytes())?;
        }
        emitted += count as u64;
    }
    writer.write_all(&0_u32.to_le_bytes())?;
    writer.flush()?;
    eprintln!(
        "meteo-mlx-native evaluated {emitted} quantised positions with {}",
        network_path.display()
    );
    Ok(())
}

fn tensor_f32(
    tensors: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>, AnyError> {
    let tensor = tensors.tensor(name)?;
    if tensor.dtype() != Dtype::F32 || tensor.shape() != shape {
        return Err(invalid(format!(
            "tensor {name} has dtype/shape {:?}/{:?}, expected F32/{shape:?}",
            tensor.dtype(),
            tensor.shape()
        )));
    }
    let data = tensor.data();
    if !data.len().is_multiple_of(4) {
        return Err(invalid(format!("tensor {name} has a partial f32 payload")));
    }
    let values = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk is four bytes")))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!("tensor {name} contains NaN or infinity")));
    }
    Ok(values)
}

fn export(options: &BTreeMap<String, String>) -> Result<(), AnyError> {
    check_exact_options(
        options,
        &["model", "tatara-output", "yaneuraou-output", "fv-scale"],
    )?;
    let model_path = required_path(options, "model")?;
    let tatara_path = required_path(options, "tatara-output")?;
    let yaneuraou_path = required_path(options, "yaneuraou-output")?;
    if tatara_path == yaneuraou_path || model_path == tatara_path || model_path == yaneuraou_path {
        return Err(invalid("model and export paths must all be different"));
    }
    let fv_scale = required(options, "fv-scale")?.parse::<i32>()?;
    if fv_scale <= 0 {
        return Err(invalid("fv-scale must be positive"));
    }
    let bytes = fs::read(&model_path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut ft_w = tensor_f32(&tensors, "ft_real", &[FT_IN, FT_OUT])?;
    let ft_virtual = tensor_f32(&tensors, "ft_virtual", &[PIECE_INPUTS, FT_OUT])?;
    for feature in 0..FT_IN {
        let real_offset = feature * FT_OUT;
        let virtual_offset = (feature % PIECE_INPUTS) * FT_OUT;
        for output in 0..FT_OUT {
            ft_w[real_offset + output] += ft_virtual[virtual_offset + output];
        }
    }
    let weights = LayerStackWeights {
        feature_set: FeatureSet::HalfKaHmMerged.spec(),
        num_buckets: NUM_BUCKETS,
        ft_w,
        ft_b: tensor_f32(&tensors, "ft_b", &[FT_OUT])?,
        l1_w: tensor_f32(&tensors, "l1_w", &[NUM_BUCKETS, L1_OUT, FT_OUT])?,
        l1_b: tensor_f32(&tensors, "l1_b", &[NUM_BUCKETS, L1_OUT])?,
        l1f_w: tensor_f32(&tensors, "l1f_w", &[FT_OUT, L1_OUT])?,
        l1f_b: tensor_f32(&tensors, "l1f_b", &[L1_OUT])?,
        l2_w: tensor_f32(&tensors, "l2_w", &[NUM_BUCKETS, L2_OUT, L2_IN])?,
        l2_b: tensor_f32(&tensors, "l2_b", &[NUM_BUCKETS, L2_OUT])?,
        l3_w: tensor_f32(&tensors, "l3_w", &[NUM_BUCKETS, L2_OUT])?,
        l3_b: tensor_f32(&tensors, "l3_b", &[NUM_BUCKETS])?,
        psqt_w: None,
    };
    let tatara_file = File::create(&tatara_path)?;
    let mut tatara_writer = BufWriter::new(tatara_file);
    weights.save_quantised(&mut tatara_writer, Some(fv_scale))?;
    tatara_writer.flush()?;
    let yaneuraou_file = File::create(&yaneuraou_path)?;
    let mut yaneuraou_writer = BufWriter::new(yaneuraou_file);
    save_yaneuraou(&mut yaneuraou_writer, &weights)?;
    yaneuraou_writer.flush()?;
    eprintln!(
        "meteo-mlx-native exported {} and {}",
        tatara_path.display(),
        yaneuraou_path.display()
    );
    Ok(())
}

fn run() -> Result<(), AnyError> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let Some(command) = args.first() else {
        return Err(invalid(
            "usage: meteo-mlx-native <stream|initialise|export|evaluate-quantised> [--name value ...]",
        ));
    };
    let options = parse_options(&args[1..])?;
    match command.as_str() {
        "stream" => stream(&options),
        "initialise" => initialise(&options),
        "export" => export(&options),
        "evaluate-quantised" => evaluate_quantised(&options),
        other => Err(invalid(format!("unknown command {other}"))),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("meteo-mlx-native error: {error}");
        std::process::exit(2);
    }
}
