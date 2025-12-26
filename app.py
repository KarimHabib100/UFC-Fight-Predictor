"""
UFC Fight Predictor - Streamlit Web Application

A professional web interface for predicting UFC fight outcomes.

Features:
- Single fight predictions with explanations
- Fighter search and statistics
- Fight card predictions
- Model performance dashboard
- SHAP explainability visualizations

Run with: streamlit run app.py

Author: UFC Fight Predictor Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.prediction.predict import FightPredictor, print_prediction

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #D20A0A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .winner-red {
        color: #D20A0A;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .winner-blue {
        color: #1E88E5;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .confidence-high {
        color: #4CAF50;
    }
    .confidence-medium {
        color: #FF9800;
    }
    .confidence-low {
        color: #F44336;
    }
    .stat-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# LOAD PREDICTOR (cached)
# =============================================================================

@st.cache_resource
def load_predictor():
    """Load the fight predictor (cached for performance)."""
    return FightPredictor(model_name='lightgbm')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level."""
    if confidence >= 0.7:
        return "#4CAF50"  # Green
    elif confidence >= 0.6:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red


def create_probability_gauge(red_prob: float, red_name: str, blue_name: str) -> go.Figure:
    """Create a probability gauge chart."""
    fig = go.Figure()

    # Red bar
    fig.add_trace(go.Bar(
        x=[red_prob * 100],
        y=['Probability'],
        orientation='h',
        name=red_name,
        marker_color='#D20A0A',
        text=[f'{red_prob:.1%}'],
        textposition='inside',
        insidetextanchor='middle'
    ))

    # Blue bar
    fig.add_trace(go.Bar(
        x=[(1 - red_prob) * 100],
        y=['Probability'],
        orientation='h',
        name=blue_name,
        marker_color='#1E88E5',
        text=[f'{1-red_prob:.1%}'],
        textposition='inside',
        insidetextanchor='middle'
    ))

    fig.update_layout(
        barmode='stack',
        height=100,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1, xanchor='center', x=0.5),
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False)
    )

    return fig


def create_comparison_radar(red_stats: dict, blue_stats: dict,
                           red_name: str, blue_name: str) -> go.Figure:
    """Create a radar chart comparing two fighters."""
    categories = ['Wins', 'KO Power', 'Striking', 'Grappling', 'Experience']

    # Normalize stats (simplified)
    red_values = [
        min(red_stats.get('Wins', 0) / 3, 1),  # Normalize wins
        min(red_stats.get('WinsByKO', 0) / 2, 1),  # KO power
        min((red_stats.get('AvgSigStrLanded', 0) + 1) / 2, 1),  # Striking
        min((red_stats.get('AvgTDLanded', 0) + 1) / 2, 1),  # Grappling
        min(red_stats.get('TotalRoundsFought', 0) / 3, 1)  # Experience
    ]

    blue_values = [
        min(blue_stats.get('Wins', 0) / 3, 1),
        min(blue_stats.get('WinsByKO', 0) / 2, 1),
        min((blue_stats.get('AvgSigStrLanded', 0) + 1) / 2, 1),
        min((blue_stats.get('AvgTDLanded', 0) + 1) / 2, 1),
        min(blue_stats.get('TotalRoundsFought', 0) / 3, 1)
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=red_values + [red_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=red_name,
        line_color='#D20A0A',
        fillcolor='rgba(210, 10, 10, 0.3)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=blue_values + [blue_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=blue_name,
        line_color='#1E88E5',
        fillcolor='rgba(30, 136, 229, 0.3)'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=400
    )

    return fig


def create_shap_bar_chart(explanation: dict, red_name: str, blue_name: str) -> go.Figure:
    """Create a bar chart of SHAP values."""
    factors_red = explanation.get('top_factors_red', [])
    factors_blue = explanation.get('top_factors_blue', [])

    features = []
    values = []
    colors = []

    for f in factors_red[:5]:
        features.append(f['feature'])
        values.append(f['impact'])
        colors.append('#D20A0A')

    for f in factors_blue[:5]:
        features.append(f['feature'])
        values.append(-f['impact'])
        colors.append('#1E88E5')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=features,
        x=values,
        orientation='h',
        marker_color=colors
    ))

    fig.update_layout(
        title='Key Prediction Factors',
        xaxis_title='Impact on Prediction',
        yaxis_title='Feature',
        height=400,
        showlegend=False
    )

    # Add center line
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    # Add annotations
    fig.add_annotation(x=0.1, y=1.1, text=f"Favors {red_name}",
                      showarrow=False, xref="paper", yref="paper",
                      font=dict(color='#D20A0A'))
    fig.add_annotation(x=0.9, y=1.1, text=f"Favors {blue_name}",
                      showarrow=False, xref="paper", yref="paper",
                      font=dict(color='#1E88E5'))

    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.markdown("# 🥊 UFC Predictor")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🎯 Fight Prediction", "👤 Fighter Search", "📋 Fight Card", "📊 Model Info"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This app uses machine learning to predict UFC fight outcomes.

    **Model:** LightGBM
    **Accuracy:** ~67%
    **Features:** 108
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("*Built with Streamlit*")

    return page


# =============================================================================
# PAGES
# =============================================================================

def page_fight_prediction(predictor: FightPredictor):
    """Render the fight prediction page."""
    st.markdown('<h1 class="main-header">🥊 UFC Fight Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter fighter details to get a prediction</p>', unsafe_allow_html=True)

    # Fighter selection
    col1, col2 = st.columns(2)

    fighters = predictor.get_fighter_names()

    with col1:
        st.markdown("### 🔴 Red Corner")
        red_fighter = st.selectbox(
            "Select Fighter",
            options=fighters,
            index=fighters.index('Jon Jones') if 'Jon Jones' in fighters else 0,
            key='red_fighter'
        )
        red_odds = st.number_input(
            "American Odds",
            value=-200,
            min_value=-1000,
            max_value=1000,
            step=10,
            key='red_odds'
        )

    with col2:
        st.markdown("### 🔵 Blue Corner")
        blue_fighter = st.selectbox(
            "Select Fighter",
            options=fighters,
            index=fighters.index('Stipe Miocic') if 'Stipe Miocic' in fighters else 1,
            key='blue_fighter'
        )
        blue_odds = st.number_input(
            "American Odds",
            value=170,
            min_value=-1000,
            max_value=1000,
            step=10,
            key='blue_odds'
        )

    # Fight details
    st.markdown("### Fight Details")
    col3, col4, col5 = st.columns(3)

    with col3:
        weight_class = st.selectbox(
            "Weight Class",
            options=list(WEIGHT_CLASSES.keys()),
            index=8  # Heavyweight
        )

    with col4:
        title_bout = st.checkbox("Title Fight", value=True)

    with col5:
        num_rounds = st.selectbox("Rounds", options=[3, 5], index=1 if title_bout else 0)

    # Predict button
    st.markdown("---")

    if st.button("🎯 Predict Fight Outcome", type="primary", use_container_width=True):
        with st.spinner("Analyzing fight..."):
            result = predictor.predict(
                red_fighter=red_fighter,
                blue_fighter=blue_fighter,
                weight_class=weight_class,
                title_bout=title_bout,
                num_rounds=num_rounds,
                red_odds=red_odds,
                blue_odds=blue_odds,
                explain=True
            )

        # Display results
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")

        # Winner announcement
        winner = result['predicted_winner']
        confidence = result['confidence']
        is_red_winner = winner == red_fighter

        winner_class = "winner-red" if is_red_winner else "winner-blue"
        conf_class = "confidence-high" if confidence >= 0.7 else ("confidence-medium" if confidence >= 0.6 else "confidence-low")

        st.markdown(f"""
        <div class="prediction-box" style="text-align: center;">
            <h2>Predicted Winner</h2>
            <p class="{winner_class}">{winner}</p>
            <p class="{conf_class}">Confidence: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability gauge
        st.plotly_chart(
            create_probability_gauge(result['red_win_probability'], red_fighter, blue_fighter),
            use_container_width=True
        )

        # Detailed stats
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label=f"{red_fighter} Win Probability",
                value=f"{result['red_win_probability']:.1%}"
            )

        with col2:
            st.metric(
                label=f"{blue_fighter} Win Probability",
                value=f"{result['blue_win_probability']:.1%}"
            )

        with col3:
            st.metric(
                label="Confidence Level",
                value=result['confidence_level']
            )

        # Fighter comparison radar
        red_stats = predictor.get_fighter_stats(red_fighter) or {}
        blue_stats = predictor.get_fighter_stats(blue_fighter) or {}

        if red_stats and blue_stats:
            st.markdown("### Fighter Comparison")
            st.plotly_chart(
                create_comparison_radar(red_stats, blue_stats, red_fighter, blue_fighter),
                use_container_width=True
            )

        # SHAP explanation
        if 'explanation' in result and 'error' not in result.get('explanation', {}):
            st.markdown("### 🔍 Why This Prediction?")
            st.plotly_chart(
                create_shap_bar_chart(result['explanation'], red_fighter, blue_fighter),
                use_container_width=True
            )

            # Detailed factors
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Factors Favoring {red_fighter}:**")
                for f in result['explanation'].get('top_factors_red', [])[:5]:
                    st.markdown(f"- {f['feature']}: +{f['impact']:.3f}")

            with col2:
                st.markdown(f"**Factors Favoring {blue_fighter}:**")
                for f in result['explanation'].get('top_factors_blue', [])[:5]:
                    st.markdown(f"- {f['feature']}: {f['impact']:.3f}")


def page_fighter_search(predictor: FightPredictor):
    """Render the fighter search page."""
    st.markdown('<h1 class="main-header">👤 Fighter Search</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Search for fighters and view their statistics</p>', unsafe_allow_html=True)

    # Search box
    search_query = st.text_input("🔍 Search Fighter", placeholder="Enter fighter name...")

    if search_query:
        matches = predictor.search_fighter(search_query, limit=10)

        if matches:
            st.markdown(f"### Found {len(matches)} fighters")

            selected_fighter = st.selectbox("Select a fighter", options=matches)

            if selected_fighter:
                stats = predictor.get_fighter_stats(selected_fighter)

                if stats:
                    st.markdown(f"## {selected_fighter}")
                    st.markdown("---")

                    # Key stats in columns
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        wins = stats.get('Wins', 0)
                        st.metric("Wins", f"{wins:.0f}" if wins == int(wins) else f"{wins:.2f}")

                    with col2:
                        losses = stats.get('Losses', 0)
                        st.metric("Losses", f"{abs(losses):.0f}" if losses == int(losses) else f"{abs(losses):.2f}")

                    with col3:
                        ko = stats.get('WinsByKO', 0)
                        st.metric("KO Wins", f"{ko:.0f}" if ko == int(ko) else f"{ko:.2f}")

                    with col4:
                        sub = stats.get('WinsBySubmission', 0)
                        st.metric("Submission Wins", f"{sub:.0f}" if sub == int(sub) else f"{sub:.2f}")

                    # Detailed stats
                    st.markdown("### Detailed Statistics")

                    # Create DataFrame for display
                    stat_categories = {
                        'Record': ['Wins', 'Losses', 'Draws', 'TotalTitleBouts'],
                        'Striking': ['AvgSigStrLanded', 'AvgSigStrPct'],
                        'Grappling': ['AvgTDLanded', 'AvgTDPct', 'AvgSubAtt'],
                        'Physical': ['HeightCms', 'ReachCms', 'WeightLbs'],
                        'Finish Rate': ['WinsByKO', 'WinsBySubmission', 'WinsByDecisionUnanimous']
                    }

                    for category, stat_keys in stat_categories.items():
                        with st.expander(category):
                            for key in stat_keys:
                                val = stats.get(key, 'N/A')
                                if isinstance(val, float):
                                    st.write(f"**{key}:** {val:.2f}")
                                else:
                                    st.write(f"**{key}:** {val}")
        else:
            st.warning("No fighters found matching your search.")
    else:
        # Show random fighters
        st.markdown("### Popular Fighters")
        popular = ['Jon Jones', 'Conor Mcgregor', 'Khabib Nurmagomedov',
                   'Israel Adesanya', 'Amanda Nunes', 'Valentina Shevchenko']

        cols = st.columns(3)
        for i, fighter in enumerate(popular):
            with cols[i % 3]:
                if st.button(fighter, key=f"pop_{fighter}"):
                    st.session_state['search_fighter'] = fighter


def page_fight_card(predictor: FightPredictor):
    """Render the fight card prediction page."""
    st.markdown('<h1 class="main-header">📋 Fight Card Predictions</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict outcomes for an entire fight card</p>', unsafe_allow_html=True)

    fighters = predictor.get_fighter_names()

    # Number of fights
    num_fights = st.slider("Number of fights on card", min_value=1, max_value=15, value=5)

    # Fight inputs
    fights = []

    for i in range(num_fights):
        st.markdown(f"### Fight {i + 1}")

        col1, col2, col3, col4 = st.columns([3, 2, 3, 2])

        with col1:
            red = st.selectbox(
                "Red Corner",
                options=fighters,
                key=f"card_red_{i}"
            )

        with col2:
            red_odds = st.number_input(
                "Red Odds",
                value=-150,
                min_value=-1000,
                max_value=1000,
                step=10,
                key=f"card_red_odds_{i}"
            )

        with col3:
            blue = st.selectbox(
                "Blue Corner",
                options=fighters,
                index=min(1, len(fighters)-1),
                key=f"card_blue_{i}"
            )

        with col4:
            blue_odds = st.number_input(
                "Blue Odds",
                value=130,
                min_value=-1000,
                max_value=1000,
                step=10,
                key=f"card_blue_odds_{i}"
            )

        fights.append({
            'red_fighter': red,
            'blue_fighter': blue,
            'red_odds': red_odds,
            'blue_odds': blue_odds,
            'weight_class': 'Lightweight',
            'title_bout': False,
            'num_rounds': 3
        })

        st.markdown("---")

    # Predict all
    if st.button("🎯 Predict All Fights", type="primary", use_container_width=True):
        with st.spinner("Analyzing fight card..."):
            results = predictor.predict_card(fights)

        st.markdown("## 📊 Card Predictions")

        # Results table
        table_data = []
        for i, result in enumerate(results):
            if 'error' not in result:
                table_data.append({
                    'Fight': f"{result['red_fighter']} vs {result['blue_fighter']}",
                    'Predicted Winner': result['predicted_winner'],
                    'Confidence': f"{result['confidence']:.1%}",
                    'Red Win %': f"{result['red_win_probability']:.1%}",
                    'Blue Win %': f"{result['blue_win_probability']:.1%}"
                })

        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Visual summary
        st.markdown("### Prediction Summary")

        for i, result in enumerate(results):
            if 'error' not in result:
                is_red = result['predicted_winner'] == result['red_fighter']
                color = "#D20A0A" if is_red else "#1E88E5"

                st.markdown(f"""
                <div style="background-color: {color}22; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 4px solid {color};">
                    <strong>Fight {i+1}:</strong> {result['predicted_winner']} wins
                    ({result['confidence']:.1%} confidence)
                </div>
                """, unsafe_allow_html=True)


def page_model_info(predictor: FightPredictor):
    """Render the model information page."""
    st.markdown('<h1 class="main-header">📊 Model Information</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Learn about the prediction model</p>', unsafe_allow_html=True)

    # Model overview
    st.markdown("## Model Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Type", "LightGBM")

    with col2:
        st.metric("Test Accuracy", "67.4%")

    with col3:
        st.metric("ROC-AUC", "0.724")

    # Feature information
    st.markdown("## Features Used")

    st.markdown("""
    The model uses **108 features** across these categories:

    - **Betting Features**: Odds, implied probabilities, expected value
    - **Fighter Statistics**: Win/loss records, finish rates, streaks
    - **Physical Attributes**: Height, reach, weight
    - **Performance Metrics**: Striking accuracy, takedown success, submission attempts
    - **Ranking Data**: UFC rankings, pound-for-pound rankings
    """)

    # Top features
    st.markdown("## Most Important Features (SHAP)")

    top_features = [
        ("RedExpectedValue", "Betting value for Red corner"),
        ("ImpliedProbDiff", "Difference in win probabilities"),
        ("BlueOdds", "Betting odds for Blue corner"),
        ("BlueAvgSigStrLanded", "Blue's striking output"),
        ("OddsDiff", "Difference in betting odds"),
        ("RedOdds", "Betting odds for Red corner"),
        ("BlueAvgSigStrPct", "Blue's striking accuracy"),
        ("RedAvgSigStrLanded", "Red's striking output"),
        ("RedImpliedProb", "Red's implied win probability"),
        ("RedAvgTDLanded", "Red's takedown success")
    ]

    df = pd.DataFrame(top_features, columns=['Feature', 'Description'])
    df.index = range(1, len(df) + 1)
    df.index.name = 'Rank'
    st.dataframe(df, use_container_width=True)

    # Methodology
    st.markdown("## Methodology")

    with st.expander("Data Collection"):
        st.markdown("""
        - **Source**: Ultimate UFC Dataset
        - **Fights**: 6,541 historical UFC fights
        - **Date Range**: 2010 - 2024
        - **Features**: Fighter statistics, betting odds, physical attributes
        """)

    with st.expander("Model Training"):
        st.markdown("""
        - **Algorithm**: LightGBM (Gradient Boosting)
        - **Optimization**: Optuna hyperparameter tuning
        - **Validation**: Chronological train/val/test split
        - **No Data Leakage**: Only pre-fight information used
        """)

    with st.expander("Limitations"):
        st.markdown("""
        - Predictions are probabilistic, not guarantees
        - Model doesn't account for injuries, weight cuts, or fight-night factors
        - Historical performance may not predict future results
        - Vegas odds already incorporate significant information
        """)

    # Disclaimer
    st.markdown("---")
    st.warning("""
    **Disclaimer**: This tool is for educational and entertainment purposes only.
    Do not use for gambling decisions. Past performance does not guarantee future results.
    """)


# Weight classes constant for the page
WEIGHT_CLASSES = {
    'Strawweight': 115,
    'Flyweight': 125,
    'Bantamweight': 135,
    'Featherweight': 145,
    'Lightweight': 155,
    'Welterweight': 170,
    'Middleweight': 185,
    'Light Heavyweight': 205,
    'Heavyweight': 265,
    "Women's Strawweight": 115,
    "Women's Flyweight": 125,
    "Women's Bantamweight": 135,
    "Women's Featherweight": 145
}


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    # Load predictor
    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Render sidebar and get selected page
    page = render_sidebar()

    # Render selected page
    if page == "🎯 Fight Prediction":
        page_fight_prediction(predictor)
    elif page == "👤 Fighter Search":
        page_fighter_search(predictor)
    elif page == "📋 Fight Card":
        page_fight_card(predictor)
    elif page == "📊 Model Info":
        page_model_info(predictor)


if __name__ == "__main__":
    main()
