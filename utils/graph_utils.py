import scipy
import plotly.express as px
from IPython.display import Markdown
from scipy.stats import chi2_contingency

themes = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]

def render_fig(fig, theme="simple_white"):
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True

    fig.update_layout(template=theme)
    fig.update_layout(bargap=0.35)  # 0.1 means 10% gap
    fig.show(config={'displayModeBar': False})#, renderer="png")

    
def test_corr_with_chart(temp_df, x, y, x_label, y_label, title):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(temp_df[x].astype(float), temp_df[y].astype(float))

    line = slope * temp_df[x] + intercept

    # Create scatter plot
    fig = px.scatter(temp_df.reset_index(), x=x, y=y,
                     # title='Scatter Plot with Correlation Line',
                     hover_data=['state'],
                     trendline="ols",

                                      labels={
                         x: x_label,
                         y: y_label}
                    )

    fig.update_layout(width=900, height=500, title=title)


    # Show plot
    render_fig(fig)

    # Print R-squared
    display(Markdown(f"```R² = {r_value**2:.3f}```"))

    # Print p-value
    display(Markdown(f"```p-value = {p_value:.3f}```"))

    # Interpret significance
    if p_value < 0.05:
        display(Markdown("```The relationship is statistically significant.```"))
    else:
        display(Markdown("```The relationship is not statistically significant.```"))

def test_chi_squared(table, reason, alpha=0.05):
    # Perform Chi-Squared Test
    chi2, p, dof, expected = chi2_contingency(table)

    print()
    print(table)
    print()
    # Print the result
    print(f"Chi2 value: {chi2:.2f}")
    print(f"P-value: {p:.5f}")
    print(f"Degrees of freedom: {dof:.2f}")

    # Interpret p-value
    if p < alpha:
        print(f"Reject null hypothesis: There is a relationship {reason}.")
    else:
        print(f"Fail to reject null hypothesis: There is no relationship {reason}.")
    print(f"*alpha = {alpha}")
