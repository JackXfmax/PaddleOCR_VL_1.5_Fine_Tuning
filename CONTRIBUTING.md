# Contributing to TibetanOCR-VL

First off, thank you for considering contributing! 🎉

## Ways to Contribute

### 🐛 Report Bugs
Found a bug in OCR recognition, training pipeline, or inference service?
Open an issue using the Bug Report template.

### 📊 Contribute Data
We welcome high-quality Tibetan text image annotations:
- Natural scene images with Tibetan text
- Tibetan calligraphy samples
- Multi-lingual signage (Tibetan + Chinese + English)

For data contributions, please open a Discussion first.

### 🧠 Algorithm Improvements
Ideas for improving the model architecture, training strategy, or LoRA
configuration are highly valued. Open a Discussion with your proposal
before implementing.

### 📝 Documentation
Help us improve documentation for:
- Chinese (简体中文) primary language
- English translations
- Tibetan (བོད་སྐད།) translations — highly welcome!

## Development Workflow

```bash
# 1. Fork & Clone
git clone https://github.com/your-username/TibetanOCR-VL.git
cd TibetanOCR-VL

# 2. Set up environment
bash setup_env.sh

# 3. Install dev dependencies
pip install -r requirements-dev.txt
pre-commit install

# 4. Create branch
git checkout -b feat/your-feature

# 5. Make changes + test
python -m pytest tests/

# 6. Commit (Conventional Commits)
git commit -m "feat(data): add X dataset converter"

# 7. Push & Create PR
git push origin feat/your-feature
```

## Commit Convention

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `style` | Code style (formatting) |
| `refactor` | Code refactoring |
| `perf` | Performance improvement |
| `test` | Adding tests |
| `chore` | Maintenance tasks |

## Pull Request Checklist

- [ ] Code follows project style (Black + isort + flake8)
- [ ] Tests pass locally
- [ ] Documentation updated if needed
- [ ] Linked to related Issue
- [ ] Screenshots attached (for UI changes)
- [ ] Signed-off commits (DCO)

## Questions?

Open a Discussion or contact the maintainer.
