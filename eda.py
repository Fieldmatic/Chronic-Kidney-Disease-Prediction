import seaborn as sns
import plotly.express as px

from main import df


def violin(col):
    fig = px.violin(df, y=col, x="class", color="class", box=True, template='plotly_dark')
    return fig.show()


def kde(col):
    grid = sns.FacetGrid(df, hue="class", height=6, aspect=2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()


def scatter(col1, col2):
    fig = px.scatter(df, x=col1, y=col2, color="class", template='plotly_dark')
    return fig.show()