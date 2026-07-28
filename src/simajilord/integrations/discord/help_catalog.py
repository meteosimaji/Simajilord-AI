"""Curated help text for the public Discord command surface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HelpEntry:
    topic: str
    category: str
    summary: str
    usage: str
    examples: tuple[str, ...]
    notes: tuple[str, ...] = ()


def _entry(
    topic: str,
    category: str,
    summary: str,
    usage: str,
    *examples: str,
    notes: tuple[str, ...] = (),
) -> HelpEntry:
    return HelpEntry(topic, category, summary, usage, examples, notes)


HELP_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Getting started": "Find commands and check whether the platform is healthy.",
    "Audio": "Play music, run Radio, manage the shared audio panel, and set timers.",
    "Read aloud": "Read several conversation channels into one voice channel.",
    "Web & media": "Search the web, inspect pages, save public media, and analyse uploads.",
    "Discord": "Inspect public server/member data and create native Discord content.",
}


HELP_ENTRIES: tuple[HelpEntry, ...] = (
    _entry(
        "help",
        "Getting started",
        "Browse every public command with usage, examples, and requirements.",
        "/help [topic]",
        "/help",
        "/help topic:play",
        notes=("The topic field supports autocomplete.",),
    ),
    _entry(
        "ping",
        "Getting started",
        "Check Discord gateway latency and whether the BOT can answer.",
        "/ping",
        "/ping",
    ),
    _entry(
        "status",
        "Getting started",
        "Show platform, AI, audio, read-aloud, and web-search readiness.",
        "/status",
        "/status",
    ),
    _entry(
        "uptime",
        "Getting started",
        "Show when this process started and how long it has been running.",
        "/uptime",
        "/uptime",
    ),
    _entry(
        "about",
        "Getting started",
        "Explain the Simajilord platform and this Discord entrance.",
        "/about",
        "/about",
    ),
    _entry(
        "capabilities",
        "Getting started",
        "Search the underlying capability APIs by what you want to accomplish.",
        "/capabilities [query]",
        "/capabilities query:search the web",
        "/capabilities query:play music",
        notes=("This is an API inventory; use /help for human command instructions.",),
    ),
    _entry(
        "play",
        "Audio",
        "Resolve a song, artist, or public URL and add the selected track.",
        "/play reference:<song, artist, or URL>",
        "/play reference:Good Morning World BURNOUT SYNDROMES",
        "/play reference:https://www.youtube.com/watch?v=...",
        notes=(
            "Join a voice channel to start immediately.",
            "If you are outside voice, the request waits and starts when you join.",
        ),
    ),
    _entry(
        "audio",
        "Audio",
        "Open the single music and read-aloud control panel.",
        "/audio",
        "/audio",
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
    ),
    _entry(
        "music pause",
        "Audio",
        "Pause the current track while preserving its position.",
        "/music pause",
        "/music pause",
        notes=("You must be in the same VC as the BOT.",),
    ),
    _entry(
        "music resume",
        "Audio",
        "Resume paused audio.",
        "/music resume",
        "/music resume",
        notes=("You must be in the same VC as the BOT.",),
    ),
    _entry(
        "music skip",
        "Audio",
        "Fade out the current track and continue to the next request.",
        "/music skip",
        "/music skip",
        notes=("You must be in the same VC as the BOT.",),
    ),
    _entry(
        "music stop",
        "Audio",
        "Stop playback and clear the pending music queue.",
        "/music stop",
        "/music stop",
        notes=("This affects the whole server audio session.",),
    ),
    _entry(
        "music leave",
        "Audio",
        "Disconnect the BOT from voice.",
        "/music leave",
        "/music leave",
    ),
    _entry(
        "music loop",
        "Audio",
        "Choose no loop, current-track loop, or whole-queue loop.",
        "/music loop mode:<none|track|queue>",
        "/music loop mode:track",
        "/music loop mode:none",
    ),
    _entry(
        "music remove",
        "Audio",
        "Remove one pending item by the position shown in the queue.",
        "/music remove position:<number>",
        "/music remove position:3",
    ),
    _entry(
        "music autoleave",
        "Audio",
        "Choose whether the BOT leaves after the VC is empty.",
        "/music autoleave enabled:<true|false>",
        "/music autoleave enabled:true",
    ),
    _entry(
        "music shuffle",
        "Audio",
        "Shuffle pending manual requests without changing the current track.",
        "/music shuffle",
        "/music shuffle",
    ),
    _entry(
        "music seek",
        "Audio",
        "Move within the current track.",
        "/music seek position:<seconds, mm:ss, +seconds, or -seconds>",
        "/music seek position:1:30",
        "/music seek position:+30",
    ),
    _entry(
        "music tune",
        "Audio",
        "Adjust playback speed and pitch.",
        "/music tune speed:<0.5-2.0> pitch:<0.5-2.0>",
        "/music tune speed:1.1 pitch:1.0",
    ),
    _entry(
        "music volume",
        "Audio",
        "Set music and read-aloud volume independently.",
        "/music volume [music] [read_aloud]",
        "/music volume music:70 read_aloud:110",
    ),
    _entry(
        "music move",
        "Audio",
        "Move a pending request to another queue position.",
        "/music move source:<number> destination:<number>",
        "/music move source:5 destination:2",
    ),
    _entry(
        "music clear-mine",
        "Audio",
        "Remove only the pending tracks that you requested.",
        "/music clear-mine",
        "/music clear-mine",
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
        "search",
        "Web & media",
        "Search the web and return compact, linked results.",
        "/search query:<text> [depth]",
        "/search query:Discord voice API documentation",
    ),
    _entry(
        "fetch",
        "Web & media",
        "Fetch readable text and metadata from one public URL.",
        "/fetch url:<public URL> [offset]",
        "/fetch url:https://example.com/article",
    ),
    _entry(
        "find",
        "Web & media",
        "Find text inside a fetched public page.",
        "/find url:<public URL> phrase:<needle>",
        "/find url:https://example.com phrase:license",
    ),
    _entry(
        "download",
        "Web & media",
        "Save public video or audio when the result fits Discord's upload limit.",
        "/download url:<public URL> [media_type]",
        "/download url:https://... media_type:video",
        notes=("A 30-second per-user cooldown protects the downloader.",),
    ),
    _entry(
        "detectai",
        "Web & media",
        "Ask HIVE to estimate AI-generated and deepfake likelihood for an image or video.",
        "/detectai media:<attachment>",
        "/detectai media:photo.png",
        notes=("Results are estimates, not proof.", "Audio-only detection is not enabled."),
    ),
    _entry(
        "roll",
        "Discord",
        "Roll one or more dice and show every result plus the total.",
        "/roll [dice] [sides]",
        "/roll dice:2 sides:20",
    ),
    _entry(
        "choose",
        "Discord",
        "Choose one item from a comma-separated list.",
        "/choose options:<comma-separated text>",
        "/choose options:ramen,curry,sushi",
    ),
    _entry(
        "serverinfo",
        "Discord",
        "Show public server identity, population, channels, assets, and moderation settings.",
        "/serverinfo",
        "/serverinfo",
    ),
    _entry(
        "userinfo",
        "Discord",
        "Show public account and server-membership information.",
        "/userinfo [user]",
        "/userinfo",
        "/userinfo user:@Example",
    ),
    _entry(
        "avatar",
        "Discord",
        "Open a member's current display avatar at full size.",
        "/avatar [user]",
        "/avatar user:@Example",
    ),
    _entry(
        "poll",
        "Discord",
        "Create a native Discord poll from comma-separated answers.",
        "/poll question:<text> options:<answers> [hours] [multiple]",
        "/poll question:Lunch? options:Ramen,Curry,Sushi hours:24",
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


HELP_ENTRIES_BY_TOPIC: dict[str, HelpEntry] = {
    entry.topic.casefold(): entry for entry in HELP_ENTRIES
}
