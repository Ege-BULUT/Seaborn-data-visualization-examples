import os
import time
import io
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request
from starlette.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mpld3


sns.set_theme()
app = FastAPI()


origins = [
    "null",
    "http://localhost:63342",
    "http://localhost:8000"

]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"])


# # # # # # # # # # # # # # # # # # #
# # # #    M A I N P A G E    # # # #
# # # # # # # # # # # # # # # # # # #


app.mount(
    "/home",
    StaticFiles(directory="./home", html=True),
    name="static",
)

templates = Jinja2Templates(directory="home")


@app.get("/")
async def root(request: Request):
    return RedirectResponse('/home')


@app.get("/home")
async def home(request: Request):
    return templates.TemplateResponse(
        "Home.html", {"request": request}
    )


@app.get("/seaborn")
async def root(request: Request):
    return templates.TemplateResponse(
        "Seaborn.html", {"request": request}
    )

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


# # # # # # # # # # # # # # # # # # #
# # # #  C L U S T E R M A P  # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/clustermap")
async def seaborn_test():
    # Select a subset of the networks
    df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)
    used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
    used_columns = (df.columns.get_level_values("network")
                    .astype(int)
                    .isin(used_networks))
    df = df.loc[:, used_columns]

    # Create a categorical palette to identify the networks
    network_pal = sns.husl_palette(8, s=.45)
    network_lut = dict(zip(map(str, used_networks), network_pal))

    # Convert the palette to vectors that will be drawn on the side of the matrix
    networks = df.columns.get_level_values("network")
    network_colors = pd.Series(networks, index=df.columns).map(network_lut)
    g = plt.figure()

    # Draw the full plot
    g = sns.clustermap(df.corr(), center=0, cmap="vlag",
                       row_colors=network_colors, col_colors=network_colors,
                       dendrogram_ratio=(.1, .2),
                       cbar_pos=(.02, .32, .03, .2),
                       linewidths=.75, figsize=(12, 13))

    g.ax_row_dendrogram.remove()
    with io.BytesIO() as g_bytes:
        g.savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #     C A T P L O T     # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/catplot")
async def seaborn_countplot():

    df = sns.load_dataset("titanic")
    # Countplot
    g = plt.figure()

    g = sns.catplot(x="sex", hue="survived", kind="count", data=df)
    with io.BytesIO() as g_bytes:
        g.savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # # S C A T T E R P L O T # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/scatterplot")
async def seaborn_scatterplot():

    df = sns.load_dataset("fmri")
    g = plt.figure()
    g = sns.scatterplot(x="timepoint",
                        y="signal",
                        data=df)

    with io.BytesIO() as g_bytes:
        g.get_figure().savefig(g_bytes, format='png') # eğer savefig methodu yoksa önce get_figure() çalıştır.
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #      L M P L O T      # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/lmplot")
async def seaborn_lmplot():

    df = sns.load_dataset("tips")
    g = plt.figure()

    g = sns.lmplot(x='total_bill', y='tip', data=df)
    g.set(xlabel='Bill',
           ylabel='Tip',
           title='Bill & Tips')
    with io.BytesIO() as g_bytes:
        g.savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #    D I S T P L O T    # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/displot")
async def seaborn_displot():

    df = sns.set(rc={"figure.figsize": (8, 4)});
    np.random.seed(0)
    x = np.random.randn(100)
    g = plt.figure()

    g = sns.distplot(x)

    with io.BytesIO() as g_bytes:

        g.get_figure().savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #    K D E   P L O T    # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/kdeplot")
async def seaborn_kdeplot():

    df = np.random.randn(200)
    g = plt.figure()
    g = sns.kdeplot(df)

    with io.BytesIO() as g_bytes:
        g.get_figure().savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #    R E L   P L O T    # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/relplot")
async def seaborn_relplot():
    sns.set_style("ticks")
    df = sns.load_dataset('tips')
    g = plt.figure()
    g = sns.relplot(x="total_bill", y="tip", data=df)

    with io.BytesIO() as g_bytes:
        g.savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #    B O X   P L O T    # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/boxplot")
async def seaborn_boxplot():
    sns.set_style("whitegrid")
    df = sns.load_dataset('tips')
    g = plt.figure()
    g = sns.boxplot(x='day', y='total_bill', data=df)

    with io.BytesIO() as g_bytes:
        g.get_figure().savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # # V I O L I N   P L O T # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/violinplot2")
async def seaborn_violinplot2():

    sns.set_style("whitegrid")
    df = sns.load_dataset('fmri')
    g = plt.figure()
    g = sns.violinplot(x="timepoint",
                       y="signal",
                       data=df)

    with io.BytesIO() as g_bytes:
        g.get_figure().savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response

@app.get("/seaborn/violinplot1")
async def seaborn_violinplot1():

    sns.set_style("whitegrid")
    df = sns.load_dataset('titanic')
    g = plt.figure()
    g = sns.violinplot(x="sex",
                       y="age",
                       hue="survived",
                       data=df,
                       split=True)

    with io.BytesIO() as g_bytes:
        g.get_figure().savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #  J O I N T   P L O T  # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/jointplot1")
async def seaborn_jointplot1():
    sns.set_style("whitegrid")
    df = sns.load_dataset("attention")
    g = plt.figure()
    g = sns.jointplot(x="solutions",
                      y="score",
                      kind="hex",
                      data=df)

    with io.BytesIO() as g_bytes:
        g.savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response

@app.get("/seaborn/jointplot2")
async def seaborn_jointplot2():

    sns.set_style("whitegrid")
    df = sns.load_dataset("exercise")
    g = plt.figure()
    g = sns.jointplot(x="id",
                      y="pulse",
                      kind="kde",
                      data=df)

    with io.BytesIO() as g_bytes:
        g.savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response

@app.get("/seaborn/jointplot3")
async def seaborn_jointplot3():

    sns.set_style("whitegrid")
    df = sns.load_dataset("mpg")
    g = plt.figure()
    g = sns.jointplot(x="mpg",
                      y="acceleration",
                      kind="scatter",
                      data=df)

    with io.BytesIO() as g_bytes:
        g.savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #   H I S T   P L O T   # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/histplot")
async def seaborn_histplot():

    sns.set_style("whitegrid")
    np.random.seed(1)
    df = np.random.randn(1000)
    df = pd.Series(df, name="Numerical Variable")

    g = plt.figure()
    g = sns.histplot(data=df,
                     kde=True)

    with io.BytesIO() as g_bytes:
        g.get_figure().savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #    H E A T   M A P    # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/heatmap")
async def seaborn_heatmap():
    sns.set()
    df = sns.load_dataset("flights")
    df = df.pivot("month", "year", "passengers")
    g = plt.figure()
    g = sns.heatmap(df)
    plt.title("Heatmap Flight Data")

    with io.BytesIO() as g_bytes:
        g.get_figure().savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #   P A I R   G R I D   # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/pairgrid")
async def seaborn_pairgrid():
    df = sns.load_dataset('tips')
    g = plt.figure()
    g = sns.PairGrid(df)
    g = g.map_upper(sns.scatterplot)
    g = g.map_lower(sns.kdeplot)
    g = g.map_diag(sns.kdeplot, lw=2)

    with io.BytesIO() as g_bytes:
        g.savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #    B A R   P L O T    # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/barplot")
async def seaborn_barplot():
    df = sns.load_dataset('titanic')
    g = plt.figure()
    g = sns.barplot(x='who',
                y='fare',
                hue='class',
                data=df)

    with io.BytesIO() as g_bytes:
        g.get_figure().savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #  S T R I P   P L O T  # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/striplot")
async def seaborn_striplot():
    sns.set(style='whitegrid')
    df = sns.load_dataset("tips")
    g = plt.figure()
    g = sns.stripplot(x="day", y="total_bill", data=df)

    with io.BytesIO() as g_bytes:
        g.get_figure().savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #  B O X E N   P L O T  # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/boxenplot")
async def seaborn_boxenplot():
    sns.set_theme(style="whitegrid")
    df = sns.load_dataset("tips")
    g = plt.figure()
    g = sns.boxenplot(x="day", y="total_bill", data=df)

    with io.BytesIO() as g_bytes:
        g.get_figure().savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #  R E S I D   P L O T  # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/residplot")
async def seaborn_residplot():

    df = sns.load_dataset("tips")
    g = plt.figure()

    # draw residplot
    g = sns.residplot(x="total_bill",
                  y="tip",
                  data=df)
    with io.BytesIO() as g_bytes:
        g.get_figure().savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response


# # # # # # # # # # # # # # # # # # #
# # # #  S W A R M   P L O T  # # # #
# # # # # # # # # # # # # # # # # # #


@app.get("/seaborn/swarmplot")
async def seaborn_swarmplot():

    df = sns.load_dataset("tips")
    g = plt.figure()

    g = sns.swarmplot(x ="day", y = "total_bill",
                      data = df, size = 5)
    with io.BytesIO() as g_bytes:
        g.get_figure().savefig(g_bytes, format='png')
        g_bytes.seek(0)
        response = Response(g_bytes.getvalue(), media_type='image/png')
    return response