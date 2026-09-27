# 09 — Final build

**What to build:** Once the team has answered the questionnaire, enter the real values: extent, orientation, screen model and dimensions, printer bed size, and which sides face a wall. Run the whole pipeline to produce the final package: all tile STLs, the plywood templates, the painters' color sheet, the screen-area handoff and final previews. Include a print schedule based on the number of printers and the deadline.

**Blocked by:** 06 — Tiles, sections and plywood templates, 07 — What the screen and painters need, and the team's answers to the questionnaire (external).

**Status:** blocked (waiting on questionnaire answers)

- [ ] The config holds the team's answers, with the questionnaire answer cited next to each value
- [ ] The full pipeline runs from one command with no manual steps
- [ ] All automated checks from 01, 04, 06 and 07 pass on the final config
- [ ] The print schedule lists tiles per printer and the expected finish date against the deadline
- [ ] The lip test piece from 04 has been printed and fitted to the real screen before the full print starts
