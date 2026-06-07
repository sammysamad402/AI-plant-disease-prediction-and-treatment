"""
crop_calendar.py — Crop Calendar Module with Unified PlantDoc AI Design
Month-by-month farming guide for major Indian crops.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

calendar_router = APIRouter()

CROP_DATA = {
    "wheat": {
        "emoji": "🌾", "name": "Wheat",
        "months": {
            "October": {"tasks": ["Land preparation & plowing", "Soil testing", "Seed selection & treatment"], "tip": "Best sowing window: Oct 15–Nov 15 for timely sown wheat"},
            "November": {"tasks": ["Sowing (120–125 kg/ha)", "Basal fertilizer application", "First irrigation at crown root stage"], "tip": "Use certified seeds. Avoid late sowing for better yield."},
            "December": {"tasks": ["Weed management", "2nd irrigation at tillering", "Monitor for yellow rust"], "tip": "Watch for aphids in warm spells. Apply foliar spray if needed."},
            "January": {"tasks": ["3rd irrigation at jointing", "Top dressing with nitrogen (urea)", "Spray fungicide if rust detected"], "tip": "Critical irrigation period — do not skip jointing-stage watering."},
            "February": {"tasks": ["4th irrigation at booting", "Pest scouting for Hessian fly", "Apply pesticide if aphids exceed threshold"], "tip": "Check for powdery mildew symptoms on upper leaves."},
            "March": {"tasks": ["5th irrigation at grain filling", "Avoid heavy irrigation to prevent lodging", "Harvest planning"], "tip": "Stop irrigation 2 weeks before harvest for best grain quality."},
            "April": {"tasks": ["Harvesting when moisture <14%", "Threshing and drying", "Storage with proper ventilation"], "tip": "Harvest promptly after maturity to avoid shattering losses."},
        }
    },
    "rice": {
        "emoji": "🌾", "name": "Rice (Paddy)",
        "months": {
            "May": {"tasks": ["Nursery bed preparation", "Seed soaking & germination", "Land leveling for main field"], "tip": "Use 20–25 kg seed per ha for nursery. Treat seeds with fungicide."},
            "June": {"tasks": ["Transplanting (21–25 day seedlings)", "Puddling and leveling", "Basal fertilizer at transplanting"], "tip": "June rains help establish crop. Ensure 2–5 cm standing water."},
            "July": {"tasks": ["Weed management (hand weeding or herbicide)", "Top dressing nitrogen at tillering", "Monitor for blast & BLB disease"], "tip": "Keep 5 cm water level to suppress weeds naturally."},
            "August": {"tasks": ["2nd nitrogen top dressing at panicle initiation", "Pest monitoring (stem borer, BPH)", "Water management"], "tip": "Drain field for 7 days at mid-tillering to improve root development."},
            "September": {"tasks": ["Water management at grain filling", "Pest & disease scouting", "Avoid water stress at milky stage"], "tip": "Critical stage — ensure no water stress during grain filling."},
            "October": {"tasks": ["Drain field 10–15 days before harvest", "Harvest at 80% grain maturity", "Threshing and drying"], "tip": "Harvest at correct moisture (20–22%) to reduce breakage."},
        }
    },
    "cotton": {
        "emoji": "🌿", "name": "Cotton",
        "months": {
            "April": {"tasks": ["Land preparation & deep plowing", "Soil testing", "Seed selection (Bt hybrid)"], "tip": "Deep plow once in 3 years to break hard pan layer."},
            "May": {"tasks": ["Sowing (2.5–3 kg seeds/ha)", "Apply basal fertilizer", "Install irrigation system"], "tip": "Sow after soil temperature reaches 18°C consistently."},
            "June": {"tasks": ["Thinning and gap filling", "Weed management", "Monitor for early sucking pests"], "tip": "Control jassids and whiteflies early to prevent disease spread."},
            "July": {"tasks": ["Irrigation at 10–14 day intervals", "Apply nitrogen top dressing", "Scout for bollworm eggs"], "tip": "Install pheromone traps for bollworm monitoring."},
            "August": {"tasks": ["Foliar spray for micronutrients", "Bollworm management", "Topping at 65–70 days (optional)"], "tip": "Spray neem oil (3%) or spinosad for organic bollworm control."},
            "September": {"tasks": ["Reduce irrigation frequency", "Continue bollworm monitoring", "Watch for boll rot in rainy weather"], "tip": "Avoid waterlogging — ensure field drainage is functioning."},
            "October": {"tasks": ["First picking (open bolls)", "Post-picking fertilizer spray", "2nd picking after 20–25 days"], "tip": "Pick in cool morning hours for better fibre quality."},
            "November": {"tasks": ["Final picking", "Uprooting stalks for sanitation", "Field preparation for rabi crop"], "tip": "Destroy old stalks to break pest and disease cycle."},
        }
    },
    "tomato": {
        "emoji": "🍅", "name": "Tomato",
        "months": {
            "June": {"tasks": ["Nursery raising in trays", "Soil solarization of main field", "Arrange drip irrigation"], "tip": "Raise nursery in protected conditions for vigorous seedlings."},
            "July": {"tasks": ["Transplanting (25–30 day seedlings)", "Mulching with black polythene", "Apply basal dose fertilizer"], "tip": "Transplant on cloudy days or late evening to reduce stress."},
            "August": {"tasks": ["Staking and training", "Fertigation every 5–7 days", "Monitor for early blight and TLCV"], "tip": "Remove whitefly-infected leaves immediately — they spread TLCV virus."},
            "September": {"tasks": ["Foliar spray for calcium & boron", "Pest management (fruit borer)", "Pinch off lower suckers"], "tip": "Calcium spray prevents blossom end rot at fruiting stage."},
            "October": {"tasks": ["Harvest when 50% fruit turns red", "Grading and packaging", "Continue fertigation for next flush"], "tip": "Harvest every 3–4 days during peak season. Handle gently."},
            "November": {"tasks": ["Final harvest", "Plant removal and field sanitization", "Plan rabi sowing"], "tip": "Remove plant debris to reduce soil-borne disease inoculum."},
        }
    },
    "soybean": {
        "emoji": "🌱", "name": "Soybean",
        "months": {
            "June": {"tasks": ["Seed treatment (Rhizobium + PSB inoculants)", "Land preparation", "Sowing at 45×5 cm spacing"], "tip": "Never skip Rhizobium inoculation — it fixes 60–80% of nitrogen needed."},
            "July": {"tasks": ["Weed management (manual or herbicide)", "Thinning to maintain plant stand", "Monitor germination and gap filling"], "tip": "First 30 days are critical for weed control — do not delay."},
            "August": {"tasks": ["Intercultural operation", "Spray micronutrients if yellowing seen", "Scout for girdle beetle and stem fly"], "tip": "Yellow mosaic virus spreads through whiteflies — control them early."},
            "September": {"tasks": ["Monitor for pod borer", "Foliar spray at pod filling", "Reduce irrigation before harvest"], "tip": "Stop irrigation 10 days before harvest for easy field operations."},
            "October": {"tasks": ["Harvest when leaves drop and pods are brown", "Threshing immediately after harvesting", "Dry to <12% moisture for storage"], "tip": "Delay in harvest causes shattering losses — act promptly."},
        }
    },
}

MONTHS_ORDER = ["January","February","March","April","May","June",
                "July","August","September","October","November","December"]
MONTH_COLORS = {
    "January":"#4A90D9","February":"#6BB5E8","March":"#F5A623","April":"#E8724A",
    "May":"#E85454","June":"#3AB07A","July":"#2D8A5E","August":"#2D7A50",
    "September":"#5A9E4F","October":"#C8752A","November":"#8B4513","December":"#4A6FA5"
}


@calendar_router.get("/crop-calendar")
async def crop_calendar_page():
    return HTMLResponse(content=_calendar_html())


def _calendar_html() -> str:
    from shared_ui import shared_head, build_nav, TOAST_SCRIPT

    crop_buttons = "\n".join(
        f'<button class="crop-chip" onclick="showCrop(\'{key}\')">{v["emoji"]} {v["name"]}</button>'
        for key, v in CROP_DATA.items()
    )

    crop_js_data = {}
    for key, crop in CROP_DATA.items():
        months_list = []
        for month in MONTHS_ORDER:
            if month in crop["months"]:
                m = crop["months"][month]
                months_list.append({
                    "month": month,
                    "color": MONTH_COLORS.get(month, "#2E8B57"),
                    "tasks": m["tasks"],
                    "tip": m.get("tip", "")
                })
        crop_js_data[key] = {
            "name": crop["name"],
            "emoji": crop["emoji"],
            "months": months_list
        }

    import json
    js_data = json.dumps(crop_js_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>{shared_head("Crop Calendar")}<style>
.crop-selector-bar{{background:var(--p0);padding:1.25rem 1.5rem;}}
.crop-selector-inner{{max-width:1320px;margin:0 auto;}}
.crop-label{{font-size:.78rem;font-weight:700;color:rgba(255,255,255,.7);
  text-transform:uppercase;letter-spacing:.06em;margin-bottom:.65rem;}}
.crop-chips{{display:flex;flex-wrap:wrap;gap:.5rem;}}
.crop-chip{{padding:.5rem 1.1rem;border-radius:25px;border:2px solid rgba(255,255,255,.25);
  background:rgba(255,255,255,.1);color:#fff;font-size:.83rem;font-weight:700;
  cursor:pointer;transition:all .2s;font-family:inherit;}}
.crop-chip:hover{{background:rgba(255,255,255,.2);border-color:rgba(255,255,255,.5);}}
.crop-chip.active{{background:#fff;color:var(--p0);border-color:#fff;}}

.cal-intro{{background:var(--card);border-radius:var(--r);border:1px solid var(--bd);
  padding:2rem;text-align:center;box-shadow:var(--sh);}}
.cal-intro-icon{{font-size:3rem;margin-bottom:.75rem;}}

.cal-header{{background:var(--card);border-radius:var(--r);border:1px solid var(--bd);
  padding:1.25rem 1.5rem;box-shadow:var(--sh);margin-bottom:1.5rem;
  display:flex;align-items:center;gap:1rem;}}
.cal-crop-icon{{width:56px;height:56px;background:linear-gradient(135deg,var(--p1),var(--p2));
  border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.75rem;}}
.cal-crop-name{{font-size:1.35rem;font-weight:800;color:var(--tx);}}
.cal-crop-sub{{font-size:.83rem;color:var(--tx2);margin-top:.15rem;}}

.months-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1.25rem;}}

.month-card{{background:var(--card);border-radius:var(--r);border:1px solid var(--bd);
  overflow:hidden;box-shadow:var(--sh);transition:all .2s;}}
.month-card:hover{{transform:translateY(-3px);box-shadow:var(--shm);}}
.month-header{{padding:.85rem 1.1rem;color:#fff;display:flex;align-items:center;gap:.6rem;}}
.month-name{{font-size:1rem;font-weight:800;}}
.month-count{{font-size:.73rem;opacity:.85;margin-left:auto;
  background:rgba(255,255,255,.2);border-radius:10px;padding:.1rem .5rem;}}
.month-body{{padding:1rem 1.1rem;}}
.task-item{{display:flex;align-items:flex-start;gap:.5rem;padding:.35rem 0;
  border-bottom:1px solid var(--bd2);font-size:.84rem;color:var(--tx2);}}
.task-item:last-of-type{{border-bottom:none;}}
.task-bullet{{width:8px;height:8px;border-radius:50%;background:var(--p2);
  flex-shrink:0;margin-top:.35rem;}}
.month-tip{{background:var(--bg2);border-left:3px solid var(--ac);border-radius:0 8px 8px 0;
  padding:.6rem .85rem;margin-top:.85rem;font-size:.8rem;color:var(--tx2);line-height:1.5;}}
.tip-label{{font-size:.7rem;font-weight:700;color:var(--ac2);text-transform:uppercase;
  letter-spacing:.04em;margin-bottom:.2rem;}}

#calContent{{display:none;}}
</style>
</head>
<body>
{build_nav("/crop-calendar")}

<div class="pda-hero">
  <div class="pda-hero-inner">
    <h1>🗓️ Crop Calendar</h1>
    <p>Month-by-month farming guide — sowing, irrigation, fertilization, and harvest schedules</p>
    <div class="hero-badges">
      <span class="hero-badge">🌾 5 Major Crops</span>
      <span class="hero-badge">📅 12-Month Guide</span>
      <span class="hero-badge">💡 Expert Tips</span>
    </div>
  </div>
</div>

<!-- Crop selector -->
<div class="crop-selector-bar">
  <div class="crop-selector-inner">
    <div class="crop-label">Select a crop to view its seasonal calendar:</div>
    <div class="crop-chips" id="cropChips">{crop_buttons}</div>
  </div>
</div>

<div class="pda-container">

  <!-- Default intro -->
  <div id="calIntro" class="cal-intro">
    <div class="cal-intro-icon">🌱</div>
    <h2 style="font-size:1.25rem;font-weight:800;margin-bottom:.5rem;">Select a Crop Above</h2>
    <p style="font-size:.9rem;color:var(--tx2);">Choose wheat, rice, cotton, tomato, or soybean to see a detailed month-by-month farming guide with tasks and expert tips.</p>
  </div>

  <!-- Calendar content -->
  <div id="calContent">
    <div class="cal-header" id="calHeader"></div>
    <div class="months-grid" id="monthsGrid"></div>
  </div>
</div>

{TOAST_SCRIPT}
<script>
const CROPS = {js_data};

function showCrop(key) {{
  const crop = CROPS[key];
  if(!crop) return;

  // Update chip active state
  document.querySelectorAll('.crop-chip').forEach(c => c.classList.remove('active'));
  event.target.classList.add('active');

  // Hide intro, show calendar
  document.getElementById('calIntro').style.display = 'none';
  document.getElementById('calContent').style.display = 'block';

  // Header
  document.getElementById('calHeader').innerHTML = `
    <div class="cal-crop-icon">${{crop.emoji}}</div>
    <div>
      <div class="cal-crop-name">${{crop.name}} Calendar</div>
      <div class="cal-crop-sub">${{crop.months.length}} active months · Tap any month card for details</div>
    </div>
    <div style="margin-left:auto;display:flex;gap:.5rem;flex-wrap:wrap;">
      <span class="badge badge-green">${{crop.months.length}} Months</span>
    </div>`;

  // Month cards
  document.getElementById('monthsGrid').innerHTML = crop.months.map(m => `
    <div class="month-card">
      <div class="month-header" style="background:${{m.color}};">
        <div>
          <div class="month-name">${{m.month}}</div>
        </div>
        <span class="month-count">${{m.tasks.length}} tasks</span>
      </div>
      <div class="month-body">
        ${{m.tasks.map(t => `
          <div class="task-item">
            <div class="task-bullet"></div>
            <span>${{t}}</span>
          </div>`).join('')}}
        ${{m.tip ? `<div class="month-tip"><div class="tip-label">💡 Expert Tip</div>${{m.tip}}</div>` : ''}}
      </div>
    </div>`).join('');

  // Smooth scroll to content
  document.getElementById('calContent').scrollIntoView({{behavior:'smooth', block:'start'}});
}}
</script>
</body>
</html>"""