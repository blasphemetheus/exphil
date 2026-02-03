# Community Research Channels

This document catalogs the communities, communication channels, and resources where Melee AI research and development is discussed. Staying connected to these channels is essential for following the latest developments.

## Overview

Melee AI knowledge is distributed across:
- **Discord servers**: Primary real-time communication
- **GitHub**: Code and issue discussions
- **Social media**: Twitter/X, Reddit
- **Forums**: Smashboards (legacy)
- **Streaming**: Twitch, YouTube

---

## Discord Servers

### Slippi Discord (Primary)

**Invite**: [discord.gg/slippi](https://discord.com/invite/pPfEaW5)
**Members**: 40,000+
**Focus**: Slippi infrastructure, replays, AI data

**Key Channels**:
| Channel | Purpose |
|---------|---------|
| `#ai-ml` | AI/ML discussion, data sharing |
| `#dev-general` | Development discussion |
| `#replays` | Replay sharing |

**What You'll Find**:
- Fizzi's anonymized ranked replay dumps (100K+ games)
- Links to training data
- AI development announcements
- Technical support

**Key People**:
- **Fizzi** (@Fizzi36) - Slippi creator
- **vladfi1** - slippi-ai developer
- **UnclePunch** - Training Mode creator

> "From the Slippi discord server, there's an emulator, a large dump of ~100k anonymized human replays in a .slp file format, and a python API called libmelee."

---

### slippi-ai Discord

**Access**: Referenced in [slippi-ai README](https://github.com/vladfi1/slippi-ai)
**Focus**: Phillip/slippi-ai development specifically

**What You'll Find**:
- Discussion/feedback/support for slippi-ai
- Agent development discussion
- Training tips
- Bug reports

**Note**: The developer is hesitant to release trained agents publicly due to concerns about ranked ladder abuse.

---

### Training Mode Community Edition Discord

**Access**: Via [GitHub repo](https://github.com/AlexanderHarrison/TrainingMode-CommunityEdition)
**Focus**: UnclePunch Training Mode development

**What You'll Find**:
- New feature discussions
- Assistance with modding
- Community contributions

---

### Melee Modding Discord(s)

**Focus**: 20XX, Training Mode, general modding

**Related Projects**:
- 20XX Hack Pack
- Training Mode
- Custom stages/costumes

**Key Contributors**:
- Achilles (20XX creator)
- UnclePunch
- Punkline (ASM/Gecko codes)
- DRGN

---

## GitHub Organizations & Repos

### Project Slippi

**Organization**: [github.com/project-slippi](https://github.com/project-slippi)

**Key Repositories**:
| Repo | Purpose |
|------|---------|
| `project-slippi` | Main project, .slp spec |
| `slippi-launcher` | Desktop application |
| `Ishiiruka` | Dolphin fork |
| `slippi-js` | JavaScript SDK |

---

### AI Development Repos

| Repository | Author | Purpose |
|------------|--------|---------|
| [slippi-ai](https://github.com/vladfi1/slippi-ai) | vladfi1 | BC+RL training |
| [phillip](https://github.com/vladfi1/phillip) | vladfi1 | Original pure RL (deprecated) |
| [libmelee](https://github.com/altf4/libmelee) | altf4 | Python API |
| [SmashBot](https://github.com/altf4/SmashBot) | altf4 | Rule-based AI |

**Issue Trackers**: Good source for technical problems and solutions.

---

### Community Tools

| Repository | Purpose |
|------------|---------|
| [peppi](https://github.com/hohav/peppi) | Rust replay parser |
| [peppi-py](https://github.com/hohav/peppi-py) | Python bindings |
| [Training-Mode](https://github.com/UnclePunch/Training-Mode) | Practice modpack |
| [20XX-HACK-PACK](https://github.com/DRGN-DRC/20XX-HACK-PACK) | Training enhancement |

---

## Social Media

### Twitter/X

**Key Accounts to Follow**:

| Handle | Person | Focus |
|--------|--------|-------|
| [@Fizzi36](https://x.com/Fizzi36) | Jas Laferriere | Slippi updates |
| [@x_pilot](https://twitch.tv/x_pilot) | vladfi1 | AI development |
| (altf4) | Dan Petro | libmelee/SmashBot |

**Hashtags**:
- `#Slippi`
- `#MeleeAI`
- `#SSBM`

**What to Expect**:
- Release announcements
- Exhibition match results
- Technical insights
- Community discussion

---

### Reddit

**Primary Subreddit**: [r/SSBM](https://reddit.com/r/SSBM)

**Relevant Post Types**:
- AI exhibition match discussions
- Technical analysis threads
- New project announcements
- Community reactions

**Notable Threads** (search for):
- "Phillip AI"
- "slippi-ai"
- "Melee bot"
- "machine learning melee"

**Example**: `r/SSBM/comments/1hha2on/humanity_versus_the_machines_humanity_triumphs_in/`

---

## Streaming Platforms

### Twitch

**Key Channels**:

| Channel | Host | Content |
|---------|------|---------|
| [x_pilot](https://twitch.tv/x_pilot) | vladfi1 | AI matches, development |
| [aMSaRedYoshi](https://twitch.tv/amsayoshi) | aMSa | Pro vs AI exhibitions |

**What You'll Find**:
- Live AI vs human matches
- Development streams
- Community interaction
- Bot available to play via netplay

> "The bot is available to play via netplay on [vladfi1's] twitch channel. Due to phillip's high delay (18+ frames) and buffer donation, it should feel like playing locally at up to 300ms ping."

---

### YouTube

**Key Channels/Videos**:

| Content | Link |
|---------|------|
| Moky vs Phillip (FT10) | [youtube.com/watch?v=1kviVflqXc4](https://www.youtube.com/watch?v=1kviVflqXc4) |
| vladfi1's channel | Recordings and clips |
| Samox documentaries | Community context |

---

## Forums

### Smashboards

**URL**: [smashboards.com](https://smashboards.com/)
**Status**: Active but less central than Discord

**Key Threads**:

| Thread | Topic |
|--------|-------|
| [MeleeAI Project](https://smashboards.com/threads/meleeai-a-better-ai-for-ssbm-update-now-works-for-console.427984/) | Rule-based AI |
| [Melee Bot Resources](https://smashboards.com/threads/melee-bot-resources.488401/) | Development resources |
| [20XX Hack Pack](https://smashboards.com/threads/the-20xx-melee-training-hack-pack-v5-0-2-1-20-2023.351221/) | Training tools |
| [Training Mode](https://smashboards.com/threads/training-mode-v3-0-updated-12-26-20.456449/) | UnclePunch mod |

**Historical Value**: Many early discussions and technical insights archived here.

---

## Technical Resources

### Frame Data Sites

| Site | Focus |
|------|-------|
| [meleeframedata.com](https://meleeframedata.com/) | Character frame data |
| [ikneedata.com](https://ikneedata.com/) | Calculators, heatmaps |
| [melee-framedata.theshoemaker.de](https://melee-framedata.theshoemaker.de/) | Detailed frame data |

**IKneeData Features**:
- Trajectory calculator with many variables
- Heatmaps visualizing character options
- Animations at game speed (60fps)
- Shareable URLs

---

### Documentation Sites

| Site | Purpose |
|------|---------|
| [meleelibrary.com](https://www.meleelibrary.com/) | Tech skill guides, resources |
| [libmelee.readthedocs.io](https://libmelee.readthedocs.io/) | libmelee API docs |
| [SmashWiki](https://www.ssbwiki.com/) | General Melee knowledge |
| [Liquipedia Smash](https://liquipedia.net/smash/) | Tournament/player info |

**Melee Library**:
> "The most valuable collection of Super Smash Bros. Melee guides, technical data, and resources in the world."

---

### Data Sources

| Source | Content | Access |
|--------|---------|--------|
| Slippi Discord | Anonymized replays | Discord pinned messages |
| [ThePlayerDatabase](https://github.com/smashdata/ThePlayerDatabase) | Tournament metadata | SQLite download |
| [slippi.gg](https://slippi.gg/) | Ranked leaderboards | Web |
| [chartslp.com](https://chartslp.com/) | Global statistics | Web |

---

## Key People to Follow

### Core Infrastructure

| Person | Handle | Contribution |
|--------|--------|--------------|
| **Fizzi** | @Fizzi36 | Slippi creator, replay data provider |
| **altf4** (Dan) | GitHub | libmelee, SmashBot |
| **UnclePunch** | Patreon | Training Mode |

### AI Researchers

| Person | Handle | Contribution |
|--------|--------|--------------|
| **vladfi1** | GitHub, Twitch | slippi-ai, Phillip |
| **Eric Gu** | [ericyuegu.com](https://ericyuegu.com) | Transformer approach |
| **Bryan Chen** (bycn) | [bycn.github.io](https://bycn.github.io) | Project Nabla |

### Parsers/Tools

| Person | Handle | Contribution |
|--------|--------|--------------|
| **hohav** | GitHub | peppi (Rust parser) |
| **vinceau** | GitHub | slippi-stats |

---

## Prediction Markets

### Manifold Markets

**Market**: [Will a human player defeat the SSBM AI Phillip in 2025?](https://manifold.markets/NBAP/will-a-human-player-defeat-the-ssbm)

**Purpose**: Community betting on AI progress
**Value**: Gauge community expectations about AI capabilities

---

## Blogs & Personal Sites

| Site | Author | Content |
|------|--------|---------|
| [ericyuegu.com](https://ericyuegu.com/melee-pt1) | Eric Gu | Transformer training details |
| [bycn.github.io](https://bycn.github.io/2022/08/19/project-nabla-writeup.html) | Bryan Chen | Project Nabla writeup |
| [Medium (Fizzi)](https://medium.com/@fizzi36) | Fizzi | Slippi development |

---

## Patreon & Funding

| Creator | Patreon | Project |
|---------|---------|---------|
| [Fizzi](https://www.patreon.com/fizzi) | Slippi | Infrastructure |
| [UnclePunch](https://www.patreon.com/UnclePunch) | Training Mode | Practice tools |

**Note**: Slippi and related projects are community-funded, not commercially backed.

---

## Getting Involved

### For AI Developers

1. **Join Slippi Discord** → `#ai-ml` channel
2. **Star/watch** slippi-ai, libmelee on GitHub
3. **Follow** vladfi1 on Twitch for live matches
4. **Read** libmelee docs and slippi-ai README

### For Data Scientists

1. **Download** anonymized replays from Slippi Discord
2. **Use** peppi or peppi-py for parsing
3. **Reference** existing papers and blog posts
4. **Share** findings in appropriate channels

### For Contributors

1. **Check** GitHub issues for help-wanted tags
2. **Join** relevant Discord for discussion
3. **Read** existing code and documentation
4. **Submit** PRs following project guidelines

---

## Communication Norms

### Discord Etiquette

- Search before asking common questions
- Use appropriate channels
- Be respectful of developers' time
- Share findings that might help others

### GitHub Etiquette

- Check existing issues before creating new ones
- Provide reproduction steps for bugs
- Follow contribution guidelines
- Be patient with maintainers (often volunteers)

### Concerns About AI Release

> "I'm working with the right folks to mitigate [the risk that players will use AI to cheat]. That's why I'm not releasing weights currently."

Respect that some researchers don't release models to prevent competitive ladder abuse.

---

## Summary

| Category | Primary Channel | Secondary |
|----------|-----------------|-----------|
| **Real-time chat** | Slippi Discord | Project-specific Discords |
| **Code** | GitHub | - |
| **News** | Twitter/X | Reddit r/SSBM |
| **Live content** | Twitch | YouTube |
| **Documentation** | ReadTheDocs, Melee Library | SmashWiki |
| **Data** | Slippi Discord | GitHub releases |

---

## References

### Official Links
- [Slippi](https://slippi.gg/)
- [Slippi Discord](https://discord.com/invite/pPfEaW5)
- [libmelee Docs](https://libmelee.readthedocs.io/)
- [Melee Library](https://www.meleelibrary.com/)

### GitHub
- [project-slippi](https://github.com/project-slippi)
- [slippi-ai](https://github.com/vladfi1/slippi-ai)
- [libmelee](https://github.com/altf4/libmelee)
- [peppi](https://github.com/hohav/peppi)

### Social
- [r/SSBM](https://reddit.com/r/SSBM)
- [@Fizzi36](https://x.com/Fizzi36)
- [x_pilot Twitch](https://twitch.tv/x_pilot)
