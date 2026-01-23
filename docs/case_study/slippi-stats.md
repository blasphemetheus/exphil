# slippi-stats Case Study

**Repository**: https://github.com/vinceau/slippi-stats
**Website**: https://vince.id.au/slippi-stats/
**Author**: vinceau
**Status**: Active
**License**: MIT

## Overview

Browser-based Slippi stats computation tool. Instantly computes and beautifully renders statistics from .slp files with no server required—all processing happens locally.

## Key Features

- **Offline**: All computation in-browser
- **Instant**: No server delays
- **OBS Integration**: Drag-to-OBS button
- **Customizable**: Toggle stats, color schemes
- **No Download**: Pure web application

## Statistics Computed

1. **First Blood** - Who got first kill
2. **L-Cancel Accuracy** - Technical execution %
3. **Highest Damage Punish** - Max punish damage
4. **Kill Moves** - Finishing blow analysis
5. **Neutral Opener Moves** - Neutral game initiators
6. **Self-Destructs** - SD tracking

## Architecture

### Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | React 16.13 |
| Language | TypeScript (93.8%) |
| Build | Vite |
| Styling | Emotion (CSS-in-JS) + SCSS |
| Drag/Drop | React Beautiful DND |
| File Upload | React Dropzone |
| Parsing | @slippi/slippi-js 8.0.0 |

### Processing Flow

```
User drops .slp files
    ↓
React Dropzone captures
    ↓
@slippi/slippi-js parses locally
    ↓
Compute stats (lib/stats/compute.ts)
    ↓
Render results (views/)
    ↓
Optional: Drag to OBS
```

### State Management

React Context + Reducer pattern:
- `store/context.tsx` - Context definition
- `store/reducers.ts` - State transitions
- Lightweight alternative to Redux

## Code Structure

```
src/
├── components/          # UI components
│   ├── Stat/
│   ├── GameDisplay/
│   ├── HeadToHead/
│   ├── OBSDragButton/
│   └── DropPad/
├── containers/          # Smart components
│   ├── FileListInput.tsx
│   ├── StatDisplay/
│   └── StatOptions/
├── lib/
│   ├── stats/
│   │   ├── definitions/
│   │   │   ├── firstBlood.ts
│   │   │   ├── lCancelAccuracy.ts
│   │   │   ├── highestDamagePunish.ts
│   │   │   ├── killMoves.ts
│   │   │   ├── neutralOpenerMoves.ts
│   │   │   └── selfDestructs.ts
│   │   ├── compute.ts   # Core engine
│   │   └── types.ts
│   ├── demo.ts
│   └── readFile.ts
├── store/               # State management
├── views/
│   ├── MainView.tsx
│   └── RenderView.tsx
└── App.tsx
```

## Usage

1. Visit https://vince.id.au/slippi-stats/
2. Drag-and-drop .slp files
3. Click "Generate Stats"
4. Customize display
5. Drag OBS button to add browser source

## Development

```bash
npm install
npm run dev      # Vite dev server
npm run build    # Production build
```

## Extensibility

Add new stats via `StatDefinition` interface:

```typescript
// lib/stats/definitions/newStat.ts
export const newStat: StatDefinition = {
  name: "New Stat",
  compute: (games) => {
    // Return computed value
  },
  render: (value) => {
    // Return React element
  }
};
```

## Relevance to ExPhil

**Not directly applicable** - Stats tool, not AI training.

**However**:
- Shows what stats are valuable for analysis
- Demonstrates slippi-js usage patterns
- Could inform reward function design (L-cancel accuracy, neutral openers)

## References

- [Repository](https://github.com/vinceau/slippi-stats)
- [Live App](https://vince.id.au/slippi-stats/)
- [slippi-js](https://github.com/project-slippi/slippi-js)
