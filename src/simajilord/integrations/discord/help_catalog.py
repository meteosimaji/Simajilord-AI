"""Canonical metadata for the public Discord command surface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PublicCommandSpec:
    """One public command definition shared by Help and surface audits."""

    topic: str
    category: str
    summary: str
    usage: str
    examples: tuple[str, ...]
    permissions: tuple[str, ...]
    side_effects: tuple[str, ...]
    notes: tuple[str, ...] = ()
    common_errors: tuple[str, ...] = ()
    prefix_name: str | None = None


def _entry(
    topic: str,
    category: str,
    summary: str,
    usage: str,
    *examples: str,
    permissions: tuple[str, ...] = (
        "View the current channel and use application commands.",
    ),
    side_effects: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
    common_errors: tuple[str, ...] = (
        "The BOT cannot view or respond in the current channel.",
    ),
    prefix_name: str | None = None,
) -> PublicCommandSpec:
    return PublicCommandSpec(
        topic=topic,
        category=category,
        summary=summary,
        usage=usage,
        examples=examples,
        permissions=permissions,
        side_effects=side_effects,
        notes=notes,
        common_errors=common_errors,
        prefix_name=prefix_name,
    )


HELP_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Getting started": "Find commands and check whether the platform is healthy.",
    "Audio": "Play music, run Radio, manage the shared audio panel, and set timers.",
    "Read aloud": "Read several conversation channels into one voice channel.",
    "Web & media": "Search the web, inspect pages, save public media, and analyse uploads.",
    "Discord": "Inspect public server/member data and create native Discord content.",
}


PUBLIC_COMMAND_SPECS: tuple[PublicCommandSpec, ...] = (
    _entry(
        "help",
        "Getting started",
        "Browse every public command with usage, examples, and requirements.",
        "/help [topic]",
        "/help",
        "/help topic:play",
        prefix_name="help",
        notes=("The topic field supports autocomplete.",),
    ),
    _entry(
        "system ping",
        "Getting started",
        "Check Discord gateway latency and whether the BOT can answer.",
        "/system ping",
        "/system ping",
    ),
    _entry(
        "status",
        "Getting started",
        "Show platform, AI, audio, read-aloud, and web-search readiness.",
        "/status",
        "/status",
    ),
    _entry(
        "system uptime",
        "Getting started",
        "Show when this process started and how long it has been running.",
        "/system uptime",
        "/system uptime",
    ),
    _entry(
        "system about",
        "Getting started",
        "Explain the Simajilord platform and this Discord entrance.",
        "/system about",
        "/system about",
    ),
    _entry(
        "system capabilities",
        "Getting started",
        "Search the underlying capability APIs by what you want to accomplish.",
        "/system capabilities [query]",
        "/system capabilities query:search the web",
        "/system capabilities query:play music",
        notes=("This is an API inventory; use /help for human command instructions.",),
    ),
    _entry(
        "play",
        "Audio",
        "Add a song, public URL, or attached audio/video to the shared queue.",
        "/play [reference:<song, artist, or URL>] [file:<audio or video>]",
        "/play reference:Good Morning World BURNOUT SYNDROMES",
        "/play reference:https://www.youtube.com/watch?v=...",
        "/play file:recording.m4a",
        prefix_name="play",
        notes=(
            "Provide exactly one of reference or file.",
            "Join a voice channel to start immediately.",
            "If you are outside voice, the request waits and starts when you join.",
            "Use Apps → Play Audio on an existing Discord message.",
        ),
        side_effects=(
            "Adds one track to the server queue and may connect to your voice channel.",
            "Attachments are validated and stored privately for restart-safe playback.",
        ),
        common_errors=(
            "No playable result was found for the reference.",
            "The attachment is too large, too long, or has no audio stream.",
            "You are not in the BOT's voice channel.",
            "The server or per-user queue is full.",
        ),
    ),
    _entry(
        "audio",
        "Audio",
        "Open the single music and read-aloud control panel.",
        "/audio",
        "/audio",
        prefix_name="audio",
        notes=(
            "Primary controls stay visible; secondary actions are under More actions.",
            "The panel is silent and updates only when meaningful state changes.",
        ),
    ),
    _entry(
        "radio",
        "Audio",
        "Keep adding related tracks while manual requests always play first.",
        "/radio [enabled] [seeds]",
        "/radio",
        "/radio enabled:true seeds:https://youtu.be/... https://youtu.be/...",
        "/radio enabled:false",
        notes=(
            "Up to eight seed URLs may be separated by spaces.",
            "Radio and Loop cannot be active together; confirmation is requested.",
        ),
        side_effects=(
            "Changes the server Radio state and may resolve related tracks in the background.",
        ),
    ),
    _entry(
        "join",
        "Audio",
        "Connect to your VC and select up to 25 text, thread, or VC-chat sources.",
        "/join",
        "/join",
        notes=(
            "A private channel picker appears after the command.",
            "The selected sources are read into the VC you currently occupy.",
        ),
        side_effects=(
            "Creates or updates the server read-aloud route and activates voice.",
        ),
    ),
    _entry(
        "timer",
        "Audio",
        "Schedule a Discord and optional voice reminder without blocking the BOT.",
        "/timer minutes:<1-10080> [message] [voice] [focus_session]",
        "/timer minutes:25 message:Take a break voice:true",
        "/timer minutes:60 message:Focus session complete focus_session:true",
        notes=(
            "The timer survives a BOT restart.",
            "Focus session mode temporarily limits normal message read aloud.",
        ),
        side_effects=("Persists a timer and posts when it expires.",),
    ),
    _entry(
        "readaloud setup",
        "Read aloud",
        "Create or replace one managed source-to-VC route.",
        "/readaloud setup [text_channel] [voice_channel] [mode]",
        "/readaloud setup text_channel:#general voice_channel:General mode:queue",
        notes=(
            "Manage Server permission is required.",
            "Use /join when you want to select several source channels at once.",
        ),
    ),
    _entry(
        "readaloud status",
        "Read aloud",
        "Show routes, content mode, voice preset, exclusions, and announcements.",
        "/readaloud status",
        "/readaloud status",
    ),
    _entry(
        "readaloud mode",
        "Read aloud",
        "Choose messages, voice events, both, or neither.",
        "/readaloud mode mode:<all|messages|events|off>",
        "/readaloud mode mode:all",
        notes=("Manage Server permission is required.",),
    ),
    _entry(
        "readaloud dictionary",
        "Read aloud",
        "List server-specific pronunciation replacements.",
        "/readaloud dictionary",
        "/readaloud dictionary",
    ),
    _entry(
        "readaloud dictionary-add",
        "Read aloud",
        "Teach the server how a word should be pronounced.",
        "/readaloud dictionary-add word:<text> reading:<kana>",
        "/readaloud dictionary-add word:IUT reading:あいゆーてぃー",
        notes=("Manage Server permission is required.",),
    ),
    _entry(
        "readaloud dictionary-remove",
        "Read aloud",
        "Remove one server pronunciation entry.",
        "/readaloud dictionary-remove word:<text>",
        "/readaloud dictionary-remove word:IUT",
        notes=("Manage Server permission is required.",),
    ),
    _entry(
        "readaloud mute",
        "Read aloud",
        "Opt your own messages out of or back into read aloud.",
        "/readaloud mute ignored:<true|false>",
        "/readaloud mute ignored:true",
    ),
    _entry(
        "readaloud ignore-user",
        "Read aloud",
        "Exclude or restore another member's messages.",
        "/readaloud ignore-user user:<member> ignored:<true|false>",
        "/readaloud ignore-user user:@Example ignored:true",
        notes=("Manage Server permission is required.",),
    ),
    _entry(
        "readaloud ignore-role",
        "Read aloud",
        "Exclude or restore every member with a selected role.",
        "/readaloud ignore-role role:<role> ignored:<true|false>",
        "/readaloud ignore-role role:@Muted ignored:true",
        notes=("Manage Server permission is required.",),
    ),
    _entry(
        "readaloud announcements",
        "Read aloud",
        "Configure join, leave, and VC-move announcements.",
        "/readaloud announcements join:<bool> leave:<bool> move:<bool>",
        "/readaloud announcements join:true leave:true move:true",
        notes=(
            "Manage Server permission is required.",
            "Omitted switches retain their current values.",
        ),
    ),
    _entry(
        "readaloud message-style",
        "Read aloud",
        "Configure author names, replies, attachments, and VC membership filtering.",
        (
            "/readaloud message-style [author_names] [replies] "
            "[attachments] [vc_members_only]"
        ),
        (
            "/readaloud message-style author_names:true replies:true "
            "attachments:true vc_members_only:true"
        ),
        notes=(
            "Manage Server permission is required.",
            "Omitted switches retain their current values.",
        ),
    ),
    _entry(
        "readaloud server-voice",
        "Read aloud",
        "Set the default VOICEVOX voice preset for this server.",
        "/readaloud server-voice preset:<preset>",
        "/readaloud server-voice preset:calm",
        notes=("Manage Server permission is required.",),
    ),
    _entry(
        "readaloud my-voice",
        "Read aloud",
        "Choose your own voice preset or return to the server default.",
        "/readaloud my-voice preset:<preset>",
        "/readaloud my-voice preset:cute",
    ),
    _entry(
        "readaloud remove",
        "Read aloud",
        "Remove one source channel from the active route.",
        "/readaloud remove [channel]",
        "/readaloud remove channel:#links",
        notes=("When omitted, the current channel is removed.",),
    ),
    _entry(
        "readaloud disable",
        "Read aloud",
        "Disable automatic read aloud for this server.",
        "/readaloud disable",
        "/readaloud disable",
        notes=("Manage Server permission is required.",),
    ),
    _entry(
        "web search",
        "Web & media",
        "Search the web and return compact, linked results.",
        "/web search query:<text> [depth]",
        "/web search query:Discord voice API documentation",
    ),
    _entry(
        "web fetch",
        "Web & media",
        "Fetch readable text and metadata from one public URL.",
        "/web fetch url:<public URL> [offset]",
        "/web fetch url:https://example.com/article",
    ),
    _entry(
        "web find",
        "Web & media",
        "Find text inside a fetched public page.",
        "/web find url:<public URL> phrase:<needle>",
        "/web find url:https://example.com phrase:license",
    ),
    _entry(
        "translate",
        "Web & media",
        "Translate supplied or recent message text into another language.",
        "/translate target:<language> [text] [source:<language>]",
        "/translate target:English text:おはようございます",
        "/translate target:ja source:English text:Good morning",
        notes=(
            "Omit text to translate the latest visible non-BOT message.",
            "Set source only when automatic language detection is uncertain.",
            "Use Apps → Translate on a message to choose the target privately.",
            "No cloud translation API receives the text.",
        ),
    ),
    _entry(
        "media download",
        "Web & media",
        "Save public video or audio when the result fits Discord's upload limit.",
        "/media download url:<public URL> [media_type]",
        "/media download url:https://... media_type:video",
        notes=("A 30-second per-user cooldown protects the downloader.",),
    ),
    _entry(
        "media detect-ai",
        "Web & media",
        "Estimate AI-generated and deepfake likelihood for an image or video.",
        "/media detect-ai media:<attachment>",
        "/media detect-ai media:photo.png",
        notes=("Results are estimates, not proof.", "Audio-only detection is not enabled."),
    ),
    _entry(
        "utility roll",
        "Discord",
        "Roll one or more dice and show every result plus the total.",
        "/utility roll [dice] [sides]",
        "/utility roll dice:2 sides:20",
    ),
    _entry(
        "utility choose",
        "Discord",
        "Choose one item from a comma-separated list.",
        "/utility choose options:<comma-separated text>",
        "/utility choose options:ramen,curry,sushi",
    ),
    _entry(
        "info server",
        "Discord",
        "Show public server identity, population, channels, assets, and moderation settings.",
        "/info server",
        "/info server",
    ),
    _entry(
        "info user",
        "Discord",
        "Show public account and server-membership information.",
        "/info user [user]",
        "/info user",
        "/info user user:@Example",
    ),
    _entry(
        "info avatar",
        "Discord",
        "Open a member's current display avatar at full size.",
        "/info avatar [user]",
        "/info avatar user:@Example",
    ),
    _entry(
        "utility poll",
        "Discord",
        "Create a native Discord poll from comma-separated answers.",
        "/utility poll question:<text> options:<answers> [hours] [multiple]",
        "/utility poll question:Lunch? options:Ramen,Curry,Sushi hours:24",
    ),
    _entry(
        "Quote",
        "Discord",
        "Render a message as a local quote image.",
        "Message context menu → Apps → Quote",
        "Right-click a message, then choose Apps → Quote.",
        notes=("Animated emoji and stickers can be rendered when animation is enabled.",),
    ),
)


PUBLIC_COMMAND_SPECS_BY_TOPIC: dict[str, PublicCommandSpec] = {
    entry.topic.casefold(): entry for entry in PUBLIC_COMMAND_SPECS
}

# Transitional collection names for callers; both refer to the canonical specs.
HELP_ENTRIES = PUBLIC_COMMAND_SPECS
HELP_ENTRIES_BY_TOPIC = PUBLIC_COMMAND_SPECS_BY_TOPIC
