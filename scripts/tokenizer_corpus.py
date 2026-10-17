"""Tricky strings every converted tokenizer is compared on (shared by make_tokenizer_fixtures.py and verify_tokenizers.py)."""
import random

HAND = [
    "", " ", "  ", "\n", "\n\n", "\t", " \n ", "a", "A", "Hello", "Hello, world!", " Hello", "  Hello", "Hello  world", "Hello\nworld", "Hello\r\nworld",
    "The quick brown fox jumps over the lazy dog.", "THE QUICK BROWN FOX", "tHe QuIcK bRoWn", "don't I'll we're they've it's HE'S I'M",
    "12345678901234567890", "1,234.56 and 3.14159 and 1e-7", "0123 007 42", "\u0663\u0664\u0665 \u0967\u0968 \uff11\uff12\uff13",
    "caf\u00e9 na\u00efve fa\u00e7ade \u00dcber stra\u00dfe \u00e5ngstr\u00f6m", "e\u0301 a\u0308 (combining marks)", "\u00e9 vs e\u0301",
    "\u4f60\u597d\uff0c\u4e16\u754c\uff01\u4eca\u5929\u5929\u6c14\u4e0d\u9519\u3002", "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c \u30ab\u30bf\u30ab\u30ca", "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8",
    "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! \u0421\u043a\u043e\u043b\u044c\u043a\u043e \u0441\u0442\u043e\u0438\u0442?", "\u03b3\u03b5\u03b9\u03b1 \u03c3\u03bf\u03c5 \u03ba\u03cc\u03c3\u03bc\u03b5",
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645", "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd", "\u0939\u093f\u0928\u094d\u0926\u0940 \u092a\u093e\u0920",
    "emoji \U0001F600 \U0001F44D\U0001F3FD \U0001F468\u200D\U0001F469\u200D\U0001F467 \U0001F1E9\U0001F1EA \u2764\uFE0F", "\U0001F600\U0001F600\U0001F600",
    "tab\tseparated\tvalues", "trailing spaces   ", "   leading spaces", "multiple    inner    spaces", "nbsp\u00a0here and\u2003em space and zero\u200bwidth",
    "line1\nline2\r\nline3\rline4\n\n\nline7", "  \n  indented\n    more", "x" * 200, "ab" * 100, "-" * 50, "=" * 8 + "\n" + "=" * 8,
    "def f(x):\n    return x * 2  # double\n\nprint(f(21))\n",
    "{\"key\": [1, 2, {\"nested\": null}], \"s\": \"v\\n\"}", "<html><body class=\"a\">text &amp; more</body></html>",
    "https://example.com/path?q=1&r=%20z#frag", "user@example.com, +49 170 1234567", "C:\\Users\\name\\file.txt", "SELECT * FROM t WHERE a >= 10;",
    "1 + 1 = 2; x_y; a->b; a::b; #include <stdio.h>", "snake_case camelCase PascalCase kebab-case SCREAMING_CASE",
    "<s> </s> <unk> <pad> <|endoftext|> <|im_start|>user", "[INST] hi [/INST]", "\u2581 \u2581\u2581 leading \u2581 block",
    "a" + "\u0301" * 5, "\ufffd replacement", "\u0000 nul and \u0007 bell", "\u202e rtl override", "Ωμέγα ǅ ß İ ŉ",
]


def corpus(seed=1234, fuzz=200):
    rng = random.Random(seed)
    pool = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789      \n\t.,;:!?'\"()-_/\\<>@#$%^&*+={}[]|~`") + list("éüñßøåçÅÉ你好世界日本語한글привет") + ["\U0001F600", "\u00a0", "\u200b"]
    out = list(HAND)
    for _ in range(fuzz):
        n = rng.randint(1, 60)
        out.append("".join(rng.choice(pool) for _ in range(n)))
    return out


TRAIN_TEXT = [
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    "Hello, world! This is a small corpus used to train tiny tokenizers for tests.",
    "def main():\n    print('hello')\n    return 0\n",
    "Der schnelle braune Fuchs springt \u00fcber den faulen Hund. Gr\u00fc\u00dfe aus M\u00fcnchen!",
    "Le renard brun rapide saute par-dessus le chien paresseux. Caf\u00e9 cr\u00e8me, s'il vous pla\u00eet.",
    "El r\u00e1pido zorro marr\u00f3n salta sobre el perro perezoso. \u00bfC\u00f3mo est\u00e1s?",
    "\u4f60\u597d\uff0c\u4e16\u754c\uff01\u8fd9\u662f\u4e00\u4e2a\u7528\u4e8e\u6d4b\u8bd5\u7684\u5c0f\u8bed\u6599\u5e93\u3002\u4eca\u5929\u5929\u6c14\u5f88\u597d\u3002",
    "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c\u3002\u3053\u308c\u306f\u30c6\u30b9\u30c8\u7528\u306e\u30b3\u30fc\u30d1\u30b9\u3067\u3059\u3002",
    "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! \u042d\u0442\u043e \u043d\u0435\u0431\u043e\u043b\u044c\u0448\u043e\u0439 \u043a\u043e\u0440\u043f\u0443\u0441 \u0434\u043b\u044f \u0442\u0435\u0441\u0442\u043e\u0432.",
    "1234567890 3.14159 2,718.28 0042 100% $5.99 #hashtag @user",
    "{\"name\": \"value\", \"list\": [1, 2, 3], \"nested\": {\"a\": null}}",
    "<div class=\"box\">text &amp; more</div> https://example.com/a?b=c&d=e",
    "I don't think it's what they've said; we'll see. HE'S SURE I'M RIGHT.",
    "snake_case camelCase PascalCase kebab-case; a->b; x::y; if (a >= b) { return; }",
    "tabs\tand   spaces\n\nnewlines\r\nand more   spaces   here",
] * 6
