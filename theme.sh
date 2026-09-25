# nix-shell -p pywal16

find . -type f -name "H.avif" | while read -r img; do
    dir=$(dirname "$img")

    [ -f "$dir/theme.json" ] && continue

    wal -i "$img" --cols16 lighten

    jq -c '
      .colors
      | with_entries(.key |= sub("color"; ""))
    ' ~/.cache/wal/colors.json >"$dir/theme.json"
done
