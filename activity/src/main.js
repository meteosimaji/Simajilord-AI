import { DiscordSDK } from "@discord/embedded-app-sdk";
import "./style.css";

const elements = {
  artwork: document.querySelector("#artwork"),
  artist: document.querySelector("#artist"),
  connection: document.querySelector("#connection"),
  duration: document.querySelector("#duration"),
  elapsed: document.querySelector("#elapsed"),
  fatal: document.querySelector("#fatal"),
  fatalMessage: document.querySelector("#fatal-message"),
  levels: document.querySelector("#levels"),
  loop: document.querySelector("#loop"),
  mode: document.querySelector("#mode"),
  progress: document.querySelector("#progress"),
  queueCount: document.querySelector("#queue-count"),
  radio: document.querySelector("#radio"),
  readAloud: document.querySelector("#read-aloud"),
  requester: document.querySelector("#requester"),
  title: document.querySelector("#title"),
  upNext: document.querySelector("#up-next"),
};

let latest = null;
let socket = null;

function formatTime(value) {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
}

function estimatePosition(audio) {
  if (!audio?.current) return 0;
  const base = Number(audio.position_seconds) || 0;
  if (audio.paused || audio.speech_active) return base;
  return base + Math.max(0, Date.now() - Number(audio.sampled_at_ms || Date.now())) / 1000;
}

function renderProgress() {
  if (!latest?.current) {
    elements.elapsed.textContent = "0:00";
    elements.duration.textContent = "0:00";
    elements.progress.style.width = "0%";
    return;
  }
  const duration = Math.max(0, Number(latest.current.duration_seconds) || 0);
  const position = Math.min(duration || Infinity, estimatePosition(latest));
  elements.elapsed.textContent = formatTime(position);
  elements.duration.textContent = duration > 0 ? formatTime(duration) : "LIVE";
  elements.progress.style.width = duration > 0
    ? `${Math.min(100, (position / duration) * 100)}%`
    : "100%";
}

function renderQueue(items) {
  elements.upNext.replaceChildren();
  elements.queueCount.textContent = `${items.length} ${items.length === 1 ? "track" : "tracks"}`;
  if (!items.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "Manual requests appear here.";
    elements.upNext.append(empty);
    return;
  }
  items.forEach((item, index) => {
    const row = document.createElement("li");
    const number = document.createElement("span");
    const copy = document.createElement("div");
    const title = document.createElement("strong");
    const meta = document.createElement("span");
    number.className = "queue-number";
    number.textContent = String(index + 1).padStart(2, "0");
    title.textContent = item?.title || "Unknown track";
    meta.textContent = [item?.uploader, formatTime(item?.duration_seconds)]
      .filter(Boolean)
      .join(" · ");
    copy.append(title, meta);
    row.append(number, copy);
    elements.upNext.append(row);
  });
}

function render(audio) {
  latest = audio;
  document.body.classList.toggle("is-playing", Boolean(audio.current && !audio.paused));
  elements.connection.textContent = audio.connected ? "Live" : "Standby";
  elements.connection.classList.toggle("live", audio.connected);
  elements.radio.textContent = audio.radio ? "On" : "Off";
  elements.loop.textContent = audio.loop === "none" ? "Off" : audio.loop;
  elements.readAloud.textContent = audio.read_aloud ? "On" : "Off";
  elements.levels.textContent =
    `${audio.levels.music_percent} / ${audio.levels.read_aloud_percent}`;
  elements.mode.textContent = audio.speech_active
    ? "READ ALOUD"
    : audio.paused
      ? "PAUSED"
      : audio.radio
        ? "RADIO"
        : "VOICE CHANNEL";

  if (audio.current) {
    elements.title.textContent = audio.current.title;
    elements.artist.textContent = audio.current.uploader || "Unknown artist";
    elements.requester.textContent = audio.current.requested_by
      ? `Requested by ${audio.current.requested_by}`
      : "";
    if (audio.current.thumbnail_url) {
      elements.artwork.style.backgroundImage =
        `linear-gradient(145deg, transparent, rgb(0 0 0 / 45%)), url("${audio.current.thumbnail_url}")`;
    } else {
      elements.artwork.style.backgroundImage = "";
    }
  } else {
    elements.title.textContent = "No track is playing";
    elements.artist.textContent = audio.connected
      ? "Add music from the Discord audio panel."
      : "Join an active voice channel to begin.";
    elements.requester.textContent = "";
    elements.artwork.style.backgroundImage = "";
  }
  renderQueue(Array.isArray(audio.up_next) ? audio.up_next : []);
  renderProgress();
}

function showFatal(message) {
  elements.fatal.hidden = false;
  elements.fatalMessage.textContent = message;
  elements.connection.textContent = "Offline";
}

function preview() {
  render({
    sampled_at_ms: Date.now(),
    connected: true,
    paused: false,
    speech_active: false,
    position_seconds: 132,
    loop: "none",
    radio: true,
    read_aloud: true,
    levels: { music_percent: 82, read_aloud_percent: 110 },
    current: {
      title: "Primary Colors",
      uploader: "PELICAN FANCLUB",
      requested_by: "Meteo",
      duration_seconds: 274,
      thumbnail_url: null,
    },
    up_next: [
      { title: "Good Morning World!", uploader: "BURNOUT SYNDROMES", duration_seconds: 249 },
      { title: "怪獣", uploader: "サカナクション", duration_seconds: 241 },
      { title: "はてな", uploader: "PENGUIN RESEARCH", duration_seconds: 256 },
    ],
  });
}

async function connect() {
  if (new URLSearchParams(location.search).get("preview") === "1") {
    preview();
    return;
  }
  const configResponse = await fetch("/api/config", { cache: "no-store" });
  if (!configResponse.ok) throw new Error("Activity configuration is unavailable.");
  const config = await configResponse.json();
  const discordSdk = new DiscordSDK(config.client_id);
  await discordSdk.ready();

  const { code } = await discordSdk.commands.authorize({
    client_id: config.client_id,
    response_type: "code",
    state: "",
    prompt: "none",
    scope: ["identify"],
  });
  const tokenResponse = await fetch("/api/token", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code }),
  });
  if (!tokenResponse.ok) throw new Error("Discord authorization failed.");
  const token = await tokenResponse.json();
  const auth = await discordSdk.commands.authenticate({
    access_token: token.access_token,
  });
  if (!auth) throw new Error("Discord authentication failed.");
  if (!discordSdk.guildId || !discordSdk.channelId) {
    throw new Error("Launch the player from a server voice channel.");
  }

  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${protocol}//${location.host}/api/audio/ws`);
  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({
      access_token: token.access_token,
      guild_id: discordSdk.guildId,
      channel_id: discordSdk.channelId,
    }));
  });
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "audio_state") render(message.audio);
  });
  socket.addEventListener("close", (event) => {
    if (!latest) showFatal(event.reason || "The player connection closed.");
    elements.connection.textContent = "Disconnected";
    elements.connection.classList.remove("live");
  });
  socket.addEventListener("error", () => {
    if (!latest) showFatal("Could not connect to the audio session.");
  });
}

setInterval(renderProgress, 250);
setInterval(() => {
  if (socket?.readyState === WebSocket.OPEN) socket.send("refresh");
}, 15_000);

connect().catch((error) => {
  showFatal(error instanceof Error ? error.message : "Could not open the player.");
});
