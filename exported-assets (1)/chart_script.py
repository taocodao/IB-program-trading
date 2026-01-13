
import plotly.graph_objects as go

# Create figure
fig = go.Figure()

# Define the 4 sections in a 2x2 grid layout
sections = [
    {
        "title": "Research Documentation",
        "items": [
            "• 8-part quantified framework",
            "• Beta-adjusted VIX formulas",
            "• Mean reversion strategy",
            "• Risk management guidelines",
            "• Example calculations (Tesla)"
        ],
        "x": 0.25,
        "y": 0.75,
        "color": "#1FB8CD",
        "icon": "📊"
    },
    {
        "title": "Implementation Plan",
        "items": [
            "• Complete system architecture",
            "• 8 core Python modules",
            "• IB Gateway integration guide",
            "• Database schema design",
            "• 4-phase development roadmap"
        ],
        "x": 0.75,
        "y": 0.75,
        "color": "#2E8B57",
        "icon": "🏗️",
        "callout": "1365 lines of detailed specs"
    },
    {
        "title": "Code Templates & Formulas",
        "items": [
            "• All calculations ready-to-code",
            "• Data structures defined",
            "• Python implementation examples",
            "• Error handling patterns",
            "• Performance optimization tips"
        ],
        "x": 0.25,
        "y": 0.25,
        "color": "#D2BA4C",
        "icon": "💻",
        "callout": "Production-ready formulas"
    },
    {
        "title": "Configuration & Setup",
        "items": [
            "• JSON configuration template",
            "• Environment variables (.env)",
            "• Directory structure",
            "• Dependencies (requirements.txt)",
            "• IB Gateway setup instructions"
        ],
        "x": 0.75,
        "y": 0.25,
        "color": "#DB4545",
        "icon": "⚙️"
    }
]

# Add invisible scatter trace to create the canvas
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='markers',
    marker=dict(size=0.1, color='rgba(0,0,0,0)'),
    showlegend=False
))

# Add boxes and content for each section
for section in sections:
    # Add colored box
    fig.add_shape(
        type="rect",
        x0=section["x"] - 0.22,
        y0=section["y"] - 0.18,
        x1=section["x"] + 0.22,
        y1=section["y"] + 0.18,
        fillcolor=section["color"],
        opacity=0.15,
        line=dict(color=section["color"], width=2)
    )
    
    # Add icon and title
    fig.add_annotation(
        x=section["x"],
        y=section["y"] + 0.15,
        text=f"<b>{section['icon']} {section['title']}</b>",
        showarrow=False,
        font=dict(size=16, color=section["color"]),
        xanchor="center"
    )
    
    # Add items
    items_text = "<br>".join(section["items"])
    fig.add_annotation(
        x=section["x"],
        y=section["y"] - 0.02,
        text=items_text,
        showarrow=False,
        font=dict(size=11, color="#333"),
        xanchor="center",
        align="left"
    )
    
    # Add callout if exists
    if "callout" in section:
        fig.add_annotation(
            x=section["x"],
            y=section["y"] - 0.16,
            text=f"<b>✓ {section['callout']}</b>",
            showarrow=False,
            font=dict(size=10, color=section["color"]),
            xanchor="center",
            bgcolor="white",
            bordercolor=section["color"],
            borderwidth=1,
            borderpad=4
        )

# Add special callouts for key highlights
fig.add_annotation(
    x=0.75,
    y=0.58,
    text="<b>✓ Complete IB API integration guide</b>",
    showarrow=False,
    font=dict(size=10, color="#2E8B57"),
    xanchor="center",
    bgcolor="white",
    bordercolor="#2E8B57",
    borderwidth=1,
    borderpad=4
)

fig.add_annotation(
    x=0.25,
    y=0.58,
    text="<b>✓ 8 core Python modules</b>",
    showarrow=False,
    font=dict(size=10, color="#1FB8CD"),
    xanchor="center",
    bgcolor="white",
    bordercolor="#1FB8CD",
    borderwidth=1,
    borderpad=4
)

# Update layout
fig.update_layout(
    title={
        "text": "Project Deliverables Ready for Development Team<br><span style='font-size: 18px; font-weight: normal;'>Complete framework with implementation guides and templates</span>"
    },
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-0.05, 1.05]
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-0.05, 1.05]
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Save the chart
fig.write_image("deliverables_summary.png")
fig.write_image("deliverables_summary.svg", format="svg")
