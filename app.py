import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OTT Viewer Drop-off Analysis",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLE ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-box {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
        border-left: 4px solid #7c3aed;
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #a78bfa; }
    .metric-lbl { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .finding-box {
        background: #1e1e2e;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .finding-title { font-weight: 700; color: #f87171; font-size: 1rem; }
    .finding-body  { color: #cbd5e1; font-size: 0.9rem; margin-top: 6px; }
    .insight-pill {
        display: inline-block;
        background: #312e81;
        color: #c7d2fe;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        margin: 3px;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 18px;
    }
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/ott_viewer_dropoff_retention_us_v1.0.csv")
    df.drop('season_number', axis=1, errors='ignore', inplace=True)
    dialogue_map = {'low': 1, 'medium': 2, 'high': 3}
    df['dialogue_density_num'] = df['dialogue_density'].map(dialogue_map)
    df['content_complexity'] = df['cognitive_load'] * df['dialogue_density_num']
    df['engagement_score']   = df['hook_strength'] + df['pacing_score']
    df.drop('dialogue_density_num', axis=1, inplace=True)
    return df

df = load_data()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/netflix--v1.png", width=60)
st.sidebar.title("OTT Drop-off Analysis")
st.sidebar.markdown("**EDA & Data Quality Audit**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "📍 WHERE — Drop-off Location",
    "🔍 WHY — Root Cause",
    "🛡️ Data Quality Audit",
    "🎯 Recommendations"
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About**  
Abhishek Arya  
B.Tech PIE · MNNIT Allahabad  
2023 – 2027  
""")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📺 OTT Viewer Drop-off Analysis")
    st.markdown("##### EDA and Data Quality Audit on 33K+ OTT viewer-episode records")
    st.markdown("---")

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    total    = len(df)
    dropped  = df['drop_off'].sum()
    drop_pct = dropped / total * 100
    genres   = df['genre'].nunique()

    with col1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{total:,}</div><div class="metric-lbl">Total Records</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{drop_pct:.1f}%</div><div class="metric-lbl">Overall Drop-off Rate</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{dropped:,}</div><div class="metric-lbl">Viewers Who Dropped</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{genres}</div><div class="metric-lbl">Genres Analysed</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Drop-off Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df['drop_off'].value_counts()
        ax.pie(counts, labels=['Stayed (85.5%)', 'Dropped (14.5%)'],
               colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%',
               startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax.set_title('Target Variable Distribution', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("#### Drop-off Rate by Genre")
        genre_dr = df.groupby('genre')['drop_off'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#e74c3c' if v > 0.20 else '#f39c12' if v > 0.14 else '#2ecc71'
                  for v in genre_dr.values]
        bars = ax.bar(genre_dr.index, genre_dr.values * 100, color=colors, edgecolor='white')
        ax.axhline(14.5, color='black', linestyle='--', linewidth=1.5, label='Avg 14.5%')
        ax.set_ylabel('Drop-off Rate (%)')
        ax.set_title('Genre vs Drop-off Rate', fontweight='bold')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        for bar, val in zip(bars, genre_dr.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val*100:.1f}%', ha='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("#### Key Findings at a Glance")
    findings = [
        ("Drama genre has 24.2% drop-off", "8.3× higher than Comedy (2.9%). Genre is the strongest WHERE signal."),
        ("Cognitive Load 9 → 79.6% drop-off", "CL values 4/5/6 → exactly 0% drop-off. This also proves synthetic data generation."),
        ("Episode 1 → 2 is the sharpest spike", "Highest drop-off happens at the very start of the series. Episode 2 opening is critical."),
        ("Pacing score < 5 is the danger zone", "Clear threshold: below 5, drop-off stays high. Above 5, retention improves sharply."),
    ]
    for title, body in findings:
        st.markdown(f'<div class="finding-box"><div class="finding-title">🔴 {title}</div><div class="finding-body">{body}</div></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — WHERE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📍 WHERE — Drop-off Location":
    st.title("📍 Where Is Drop-off Happening?")
    st.markdown("Identifying content segments and episode positions with highest drop-off")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Genre", "Episode Number", "Dialogue Density", "Skip Intro × CL"])

    with tab1:
        st.markdown("#### Drop-off Rate by Genre")
        genre_dr = df.groupby('genre')['drop_off'].mean().sort_values(ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ['#e74c3c' if v > 0.20 else '#f39c12' if v > 0.14 else '#2ecc71'
                  for v in genre_dr.values]
        bars = axes[0].bar(genre_dr.index, genre_dr.values * 100, color=colors, edgecolor='white')
        axes[0].axhline(14.5, color='black', linestyle='--', label='Avg 14.5%')
        axes[0].set_ylabel('Drop-off Rate (%)')
        axes[0].set_title('Drop-off Rate by Genre')
        axes[0].legend()
        plt.sca(axes[0]); plt.xticks(rotation=45, ha='right')
        for bar, val in zip(bars, genre_dr.values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                         f'{val*100:.1f}%', ha='center', fontsize=8)

        genre_counts = df.groupby('genre').agg(
            total=('drop_off', 'count'),
            dropped=('drop_off', 'sum')
        ).reset_index()
        axes[1].barh(genre_counts['genre'], genre_counts['total'], color='#3498db', alpha=0.4, label='Total')
        axes[1].barh(genre_counts['genre'], genre_counts['dropped'], color='#e74c3c', label='Dropped')
        axes[1].set_title('Episode Count — Total vs Dropped')
        axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.info("**Insight:** Drama episodes show 24.2% drop-off — 8× higher than Comedy. High cognitive-load storytelling in Drama drives viewers away.")

    with tab2:
        st.markdown("#### Drop-off by Episode Number")
        ep = df.groupby('episode_number').agg(
            drop_off_rate=('drop_off', 'mean'),
            count=('drop_off', 'count')
        ).reset_index()
        ep10 = ep[ep['count'] > 50].head(12)

        fig, ax = plt.subplots(figsize=(11, 5))
        colors = ['#e74c3c' if v > 0.18 else '#f39c12' if v > 0.10 else '#2ecc71'
                  for v in ep10['drop_off_rate']]
        bars = ax.bar(ep10['episode_number'], ep10['drop_off_rate'] * 100,
                      color=colors, edgecolor='white')
        ax.axhline(14.5, color='black', linestyle='--', label='Avg 14.5%')
        ax.set_xlabel('Episode Number')
        ax.set_ylabel('Drop-off Rate (%)')
        ax.set_title('Drop-off Rate by Episode Number', fontweight='bold')
        ax.legend()
        for bar, val in zip(bars, ep10['drop_off_rate']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val*100:.1f}%', ha='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.info("**Insight:** Episode 1→2 is the sharpest spike. Viewers make their 'continue watching' decision after Episode 1. The Episode 2 hook is the most critical retention lever.")

    with tab3:
        st.markdown("#### Drop-off by Dialogue Density")
        dd_rates = df.groupby('dialogue_density')['drop_off'].mean() * 100
        cross    = pd.crosstab(df['dialogue_density'], df['drop_off'], normalize='index')

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        order = ['low', 'medium', 'high']
        vals  = [dd_rates.get(k, 0) for k in order]
        axes[0].bar(['Low', 'Medium', 'High'], vals,
                    color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='white')
        axes[0].axhline(14.5, color='black', linestyle='--', label='Avg 14.5%')
        axes[0].set_ylabel('Drop-off Rate (%)')
        axes[0].set_title('Drop-off Rate by Dialogue Density')
        axes[0].legend()
        for i, (lbl, v) in enumerate(zip(['Low', 'Medium', 'High'], vals)):
            axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

        cross_ordered = cross.reindex(order)
        cross_ordered.index = ['Low', 'Medium', 'High']
        cross_ordered.plot(kind='bar', stacked=True, ax=axes[1],
                           color=['#2ecc71', '#e74c3c'], edgecolor='white')
        axes[1].set_title('Drop-off Proportion by Dialogue Density')
        axes[1].legend(['Stayed', 'Dropped'])
        axes[1].set_xticklabels(['Low', 'Medium', 'High'], rotation=0)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.info("**Insight:** High dialogue density → 30.4% drop-off vs 6.3% for low. Too much dialogue, especially in Drama, overwhelms viewers.")

    with tab4:
        st.markdown("#### Skip Intro × Cognitive Load Interaction")
        def cl_group(x):
            if x <= 4:   return 'Low (≤4)'
            elif x <= 7: return 'Medium (5-7)'
            else:        return 'High (8-9)'
        df['cl_group']   = df['cognitive_load'].apply(cl_group)
        df['skip_label'] = df['skip_intro'].map({0: 'Watched Intro', 1: 'Skipped Intro'})

        combo = df.groupby(['skip_label', 'cl_group'])['drop_off'].mean().unstack()
        combo = combo[['Low (≤4)', 'Medium (5-7)', 'High (8-9)']]

        fig, ax = plt.subplots(figsize=(10, 5))
        x     = np.arange(3)
        width = 0.35
        ax.bar(x - width/2, combo.loc['Watched Intro'] * 100, width,
               label='Watched Intro', color='#2ecc71', edgecolor='white')
        ax.bar(x + width/2, combo.loc['Skipped Intro'] * 100, width,
               label='Skipped Intro', color='#e74c3c', edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(['Low CL', 'Medium CL', 'High CL'])
        ax.set_ylabel('Drop-off Rate (%)')
        ax.set_title('Skip Intro × Cognitive Load — Drop-off Rate', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.info("**Insight:** Skipped Intro + High Cognitive Load = most dangerous combination. Viewer missed the context AND the content is hard to follow.")

        df.drop(columns=['cl_group', 'skip_label'], errors='ignore', inplace=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WHY
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔍 WHY — Root Cause":
    st.title("🔍 Why Are Viewers Dropping Off?")
    st.markdown("Root cause analysis — which content signals drive drop-off")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Cognitive Load", "Pacing Score", "Pause Behaviour", "Correlation"])

    with tab1:
        st.markdown("#### Cognitive Load — The Strongest Single Driver")
        cl_dr = df.groupby('cognitive_load')['drop_off'].mean()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ['#e74c3c' if v > 0.5 else '#f39c12' if v > 0.1 else '#2ecc71'
                  for v in cl_dr.values]
        bars = axes[0].bar(cl_dr.index, cl_dr.values * 100, color=colors, edgecolor='white')
        axes[0].axhline(14.5, color='black', linestyle='--', label='Avg 14.5%')
        axes[0].set_xlabel('Cognitive Load Score')
        axes[0].set_ylabel('Drop-off Rate (%)')
        axes[0].set_title('Drop-off Rate by Cognitive Load')
        axes[0].legend()
        for bar, val in zip(bars, cl_dr.values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{val*100:.0f}%', ha='center', fontsize=9)

        sns.boxplot(x='drop_off', y='cognitive_load', data=df, ax=axes[1],
                    palette=['#2ecc71', '#e74c3c'])
        axes[1].set_xticklabels(['Stayed', 'Dropped'])
        axes[1].set_title('Cognitive Load Distribution — Stayed vs Dropped')
        axes[1].set_xlabel('')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        col1, col2, col3 = st.columns(3)
        col1.metric("CL 9 Drop-off", "79.6%", "Highest risk")
        col2.metric("CL 4/5/6 Drop-off", "0.0%", "Zero exceptions")
        col3.metric("CL Mean (Dropped)", f"{df[df['drop_off']==1]['cognitive_load'].mean():.1f}", "vs 5.5 for stayed")

        st.warning("**⚠️ Also a data quality flag:** CL 4/5/6 giving exactly 0.0% across 16,000+ rows is impossible in real data. This proves synthetic generation rules exist in the dataset.")

    with tab2:
        st.markdown("#### Pacing Score — Score 5 is the Retention Boundary")
        ps = df.groupby('pacing_score').agg(
            drop_off_rate=('drop_off', 'mean'),
            avg_watch=('avg_watch_percentage', 'mean')
        ).reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colors_ps = ['#e74c3c' if v < 5 else '#2ecc71' for v in ps['pacing_score']]
        axes[0].bar(ps['pacing_score'], ps['drop_off_rate'] * 100,
                    color=colors_ps, edgecolor='white')
        axes[0].axvline(4.5, color='black', linestyle='--', linewidth=2, label='Threshold: score 5')
        axes[0].set_xlabel('Pacing Score')
        axes[0].set_ylabel('Drop-off Rate (%)')
        axes[0].set_title('Drop-off Rate by Pacing Score')
        axes[0].legend()

        axes[1].plot(ps['pacing_score'], ps['avg_watch'], 'o-',
                     color='#3498db', linewidth=2)
        axes[1].axvline(4.5, color='black', linestyle='--', linewidth=2, label='Threshold: score 5')
        axes[1].set_xlabel('Pacing Score')
        axes[1].set_ylabel('Avg Watch Percentage (%)')
        axes[1].set_title('Avg Watch % by Pacing Score')
        axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.info("**Insight:** Pacing score ≥ 5 → retention improves sharply. This is an actionable production guideline: set minimum pacing score of 5 before any episode is released.")

    with tab3:
        st.markdown("#### Pause Count — Behavioural Early Warning Signal")
        pause_watch = df.groupby(['pause_count', 'drop_off'])['avg_watch_percentage'].mean().unstack()
        pause_watch.columns = ['Retained', 'Dropped']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(pause_watch.index, pause_watch['Retained'], 'o-',
                     color='#2ecc71', linewidth=2, label='Retained')
        axes[0].plot(pause_watch.index, pause_watch['Dropped'], 's-',
                     color='#e74c3c', linewidth=2, label='Dropped')
        axes[0].fill_between(pause_watch.index, pause_watch['Dropped'], pause_watch['Retained'],
                              alpha=0.1, color='grey')
        axes[0].set_xlabel('Pause Count')
        axes[0].set_ylabel('Avg Watch Percentage (%)')
        axes[0].set_title('Watch % vs Pause Count — Dropped vs Retained')
        axes[0].legend()
        axes[0].set_xticks(pause_watch.index)

        pause_dr = df.groupby('pause_count')['drop_off'].mean()
        axes[1].bar(pause_dr.index, pause_dr.values * 100,
                    color=['#2ecc71' if v < 0.1 else '#f39c12' if v < 0.3 else '#e74c3c'
                           for v in pause_dr.values],
                    edgecolor='white')
        axes[1].set_xlabel('Pause Count')
        axes[1].set_ylabel('Drop-off Rate (%)')
        axes[1].set_title('Drop-off Rate by Pause Count')
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.info("**Insight:** Pausing does NOT help — once a viewer starts struggling, they leave regardless of how many times they pause. This is a real-time churn signal.")

    with tab4:
        st.markdown("#### Correlation Heatmap")
        num_feats = ['pacing_score', 'hook_strength', 'visual_intensity',
                     'avg_watch_percentage', 'pause_count', 'rewind_count',
                     'episode_duration_min', 'cognitive_load', 'skip_intro', 'drop_off']
        corr = df[num_feats].corr()

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        corr_with_target = corr['drop_off'].drop('drop_off').sort_values(key=abs, ascending=False)
        st.markdown("**Top correlations with drop_off:**")
        cols = st.columns(5)
        for i, (feat, val) in enumerate(corr_with_target.head(5).items()):
            color = "🔴" if abs(val) > 0.3 else "🟡" if abs(val) > 0.15 else "🟢"
            cols[i].metric(feat, f"{val:.3f}", f"{color}")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATA QUALITY AUDIT
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🛡️ Data Quality Audit":
    st.title("🛡️ Data Quality Audit")
    st.markdown("Detecting leakage, synthetic patterns, and data integrity issues")
    st.markdown("---")

    st.error("**⚠️ Audit Verdict: This dataset is synthetically generated using hard if-else rules.**")
    st.markdown("Four patterns make this impossible to be real data. Each is documented below.")
    st.markdown("")

    # Finding 1
    with st.expander("🔴 Finding 1 — Cognitive Load Creates a Perfect Rule", expanded=True):
        st.markdown("**Claim:** CL values 4, 5, 6 → exactly 0% drop-off. CL 9 → 79.6% drop-off. Zero exceptions across 16,000+ rows.")
        cl_dr = df.groupby('cognitive_load').agg(
            drop_off_rate=('drop_off', 'mean'),
            count=('drop_off', 'count')
        ).reset_index()
        cl_dr['drop_off_rate_pct'] = (cl_dr['drop_off_rate'] * 100).round(1)
        st.dataframe(cl_dr[['cognitive_load', 'count', 'drop_off_rate_pct']].rename(
            columns={'cognitive_load': 'CL Score', 'count': 'Records', 'drop_off_rate_pct': 'Drop-off Rate (%)'}
        ), use_container_width=True)
        st.markdown("**Why this proves synthetic:** In any real content platform, there will always be exceptions — a viewer with CL=4 who had a bad day, a notification, a phone call. Exactly 0 exceptions across 16,000+ rows is statistically impossible.")

    # Finding 2
    with st.expander("🔴 Finding 2 — avg_watch_percentage Ranges Do Not Overlap"):
        stayed  = df[df['drop_off']==0]['avg_watch_percentage']
        dropped = df[df['drop_off']==1]['avg_watch_percentage']
        col1, col2 = st.columns(2)
        col1.metric("Stayed Range", f"{stayed.min():.0f}% — {stayed.max():.0f}%")
        col2.metric("Dropped Range", f"{dropped.min():.0f}% — {dropped.max():.0f}%")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(stayed,  bins=40, alpha=0.6, color='#2ecc71', label='Stayed')
        ax.hist(dropped, bins=40, alpha=0.6, color='#e74c3c', label='Dropped')
        ax.set_xlabel('Avg Watch Percentage')
        ax.set_ylabel('Count')
        ax.set_title('Watch % Distribution — Zero Overlap Between Groups')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("**Why this proves synthetic:** Real churn distributions always overlap. A viewer who watches 58% of an episode might drop off (got a call) or stay (episode was good). Hard cutoffs are rules, not human behaviour.")

    # Finding 3
    with st.expander("🔴 Finding 3 — retention_risk is Pure Leakage"):
        rr = pd.crosstab(df['retention_risk'], df['drop_off'])
        st.dataframe(rr, use_container_width=True)
        st.markdown("**Why this is leakage:** `retention_risk='high'` → 100% drop-off, `medium/low` → 0% drop-off. This column IS the target, written as a text label. Any model using it would show 99.9%+ accuracy — not because the model learned anything, but because it's reading the answer column.")
        st.error("**Action:** Remove `retention_risk`, `drop_off_probability`, `attention_required`, `night_watch_safe` before any modelling.")

    # Finding 4
    with st.expander("🔴 Finding 4 — attention_required and night_watch_safe"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**attention_required vs drop_off:**")
            st.dataframe(pd.crosstab(df['attention_required'], df['drop_off']), use_container_width=True)
        with col2:
            st.markdown("**night_watch_safe vs drop_off:**")
            st.dataframe(pd.crosstab(df['night_watch_safe'], df['drop_off']), use_container_width=True)
        st.markdown("`attention_required=medium` → 100% stayed (16,293 rows, 0 exceptions). `night_watch_safe=1` → 100% stayed. These are derived columns computed from the same synthetic rules.")

    st.markdown("---")
    st.markdown("#### What This Means for Modelling")
    col1, col2 = st.columns(2)
    col1.error("**With leakage columns:**\n\nROC-AUC ≈ 0.9999\n\nNot a real result. The model is reading the answer.")
    col2.success("**After removing leakage:**\n\nROC-AUC ≈ 0.75–0.85\n\nRealistic. Comparable to industry OTT churn models.")
    st.info("**Lesson:** Always audit data integrity before reporting model metrics. A 99% accuracy on a churn model should be a red flag, not a celebration.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Recommendations":
    st.title("🎯 What Should the Platform Do?")
    st.markdown("Translating EDA findings into business actions")
    st.markdown("---")

    # Watch time by risk
    st.markdown("#### Watch Time by Risk Segment")
    df['risk_segment'] = pd.cut(df['avg_watch_percentage'],
                                 bins=[0, 45, 65, 100],
                                 labels=['High Risk', 'Medium Risk', 'Low Risk'])
    risk = df.groupby('risk_segment').agg(
        avg_watch=('avg_watch_percentage', 'mean'),
        count=('drop_off', 'count'),
        drop_rate=('drop_off', 'mean')
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_r = ['#e74c3c', '#f39c12', '#2ecc71']
        bars = ax.bar(risk['risk_segment'], risk['avg_watch'],
                      color=colors_r, edgecolor='white', width=0.5)
        ax.set_ylabel('Avg Watch Percentage (%)')
        ax.set_title('Avg Watch Time by Risk Segment')
        for bar, val in zip(bars, risk['avg_watch']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(risk['risk_segment'], risk['drop_rate'] * 100,
               color=colors_r, edgecolor='white', width=0.5)
        ax.set_ylabel('Drop-off Rate (%)')
        ax.set_title('Drop-off Rate by Risk Segment')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")

    # Priority Matrix
    st.markdown("#### Initiative Priority Matrix — Impact vs Effort")
    initiatives = {
        'Add recap for high-CL episodes':  (9, 2),
        'Auto-skip intro for returning viewers': (7, 1),
        'Pause-triggered recap prompt':     (8, 3),
        'Redesign Episode 2 opening':       (9, 5),
        'Set pacing score ≥ 5 standard':   (8, 4),
        'Reduce high dialogue-density content': (7, 6),
        'CL guidelines for new shows':     (9, 7),
        'Shorten episode intros to 30s':   (6, 2),
        'Change visual intensity':          (2, 5),
        'Reduce episode duration':          (2, 6),
    }

    def q(impact, effort):
        if impact >= 7 and effort <= 4: return 'Quick Win', '#2ecc71'
        elif impact >= 7:               return 'Strategic Bet', '#3498db'
        elif effort <= 4:               return 'Consider', '#f39c12'
        else:                           return 'Deprioritise', '#e74c3c'

    fig, ax = plt.subplots(figsize=(11, 7))
    for name, (imp, eff) in initiatives.items():
        label, color = q(imp, eff)
        ax.scatter(eff, imp, s=200, color=color, zorder=5)
        ax.annotate(name, (eff, imp), textcoords="offset points",
                    xytext=(6, 5), fontsize=8.5)

    ax.axhline(7, color='grey', linestyle='--', linewidth=1)
    ax.axvline(4, color='grey', linestyle='--', linewidth=1)
    ax.text(0.5, 9.5, 'QUICK WINS',      color='#2ecc71', fontweight='bold', fontsize=11)
    ax.text(5.0, 9.5, 'STRATEGIC BETS',  color='#3498db', fontweight='bold', fontsize=11)
    ax.text(0.5, 3.2, 'CONSIDER',        color='#f39c12', fontweight='bold', fontsize=11)
    ax.text(5.0, 3.2, 'DEPRIORITISE',    color='#e74c3c', fontweight='bold', fontsize=11)
    ax.set_xlabel('Effort (1=Low → 10=High)', fontsize=11)
    ax.set_ylabel('Impact (1=Low → 10=High)', fontsize=11)
    ax.set_title('Initiative Priority Matrix', fontweight='bold', fontsize=13)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.5)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Action Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**✅ Quick Wins (Do First)**\n\n- Auto-skip intro for returning viewers\n- Recap card for high-CL episodes\n- Pause-triggered recap prompt\n- Set pacing score ≥ 5 as standard")
    with col2:
        st.info("**📌 Strategic Bets (Plan for Q2)**\n\n- Redesign Episode 2 opening\n- CL production guidelines for new shows\n- Reduce dialogue density in Drama")
    with col3:
        st.error("**❌ Deprioritise**\n\n- Changing episode duration\n- Changing visual intensity\n\nNo significant impact found in data.")

    df.drop(columns=['risk_segment'], errors='ignore', inplace=True)

