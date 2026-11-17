# Sports Predict

Simple multi-sport prediction web app (currently Soccer is fully working).

## Features
- Soccer predictions using time-weighted attack/defense model
- Auto-loads upcoming fixtures (today + 3 days)
- Manual single match prediction
- Smart paste from betting sites
- Clear LEAN recommendations + strong market flags
- Multi-sport tabs (Basketball, NFL, Baseball, Tennis – coming soon)

## How to run (Termux)

pip install -r requirements.txt
python app.py

Then open: http://localhost:8801

## Notes
- First run trains the model (takes a few minutes)
- Model is saved automatically for faster future starts
