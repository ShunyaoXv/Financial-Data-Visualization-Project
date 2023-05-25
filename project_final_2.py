#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import dash
from jupyter_dash import JupyterDash
from dash import html, dcc
from scipy.stats import norm
from dash import Input, Output, State
from datetime import date, datetime, timedelta
from statistics import NormalDist
import dash_daq as daq
import dash_mantine_components as dmc
import backtrader as bt
import warnings
import matplotlib.pyplot as plt
#warnings.filterwarnings("ignore")
import time
import collections
from backtrader_plotly.plotter import BacktraderPlotly
from backtrader_plotly.scheme import PlotScheme
import plotly.io
import plotly.express as px

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from dash.dash_table import DataTable, FormatTemplate, Format
import plotly.graph_objs as go
import time


# # function

# ## option

# s0-stock price at time 0
# 
# r- annual risk-neutral return
# 
# vol- annual stock volatility
# 
# T-days simulated
# 
# trails-num of simulation

# In[3]:


def generate_stock(s0,r,vol,T,trails=1000):
    delta_t = 1./252.   
    W = np.sqrt(delta_t) * np.random.randn(T,trails)
    ret = (r - (1/2)*vol**2) * delta_t + vol * W
    s=np.zeros([T,trails])
    s[0]=s0
    for i in range(1,T):
        s[i] = s[i-1] * np.e**ret[i]
    return s

def option_help(s,K,call):
    c=s-K
    if call:
        c[c<0]=0
    else:
        c[c>0]=0
        
    return c

# Vanilla Option
def euro_option(r,T,s,K=100,call=True):
    delta_t = 1./252.   
    c=option_help(s[-1],K,call)
        
    return np.exp(-r*T*delta_t)*c

# Binary Option
def cash_or_nothing(r,T,s,K=100,Q=1,call=True):
    delta_t = 1./252.  
    c=np.zeros(s.shape[1])
    if call:
        c[s[-1]>K]=Q
    else:
        c[s[-1]<K]=Q
    return np.exp(-r*T*delta_t)*c

def asset_or_nothing(r,T,s,K=100,Q=1,call=True):
    delta_t = 1./252.  
    c=s[-1].copy()
    if call:
        c[c<=K]=0
    else:
        c[c>=K]=0
    return np.exp(-r*T*delta_t)*c
    
#Asian Option
def average_price(r,T,s,K=100,call=True):
    delta_t = 1./252.   
    c=option_help(s.mean(0),K,call)
        
    return np.exp(-r*T*delta_t)*c

#Lookback Option
def floating_lookback(r,T,s,K=100,call=True):
    delta_t = 1./252.   
    if call:
        c=(s[-1]-s.min(0))
    else:
        c=(s.max(0)-s[-1])
        
    return np.exp(-r*T*delta_t)*c

def fixed_lookback(r,T,s,K=100,call=True):
    delta_t = 1./252. 
    if call:
        c=option_help(s.max(0),K,call)
    else:
        c=option_help(s.min(0),K,call)
        
    return np.exp(-r*T*delta_t)*c


#Barrier Option
def down_and_out(r,T,s,K=100,out=100,call=True):
    delta_t = 1./252. 
    news=s[-1][np.where((s>=out).any(axis=0))]
    
    c=np.zeros(s.shape[1])
    c[np.where((s>=out).any(axis=0))]=option_help(news,K,call)
        
    return np.exp(-r*T*delta_t)*c
    
def down_and_in(r,T,s,K=100,inn=100,call=True):  
    delta_t = 1./252. 
    news=s[-1][np.where((s<=inn).any(axis=0))]
    
    c=np.zeros(s.shape[1])
    c[np.where((s<=inn).any(axis=0))]=option_help(news,K,call)
        
    return np.exp(-r*T*delta_t)*c

def up_and_out(r,T,s,K=100,out=100,call=True):
    delta_t = 1./252. 
    news=s[-1][np.where((s<=out).any(axis=0))]
    
    c=np.zeros(s.shape[1])
    c[np.where((s<=out).any(axis=0))]=option_help(news,K,call)
        
    return np.exp(-r*T*delta_t)*c

def up_and_in(r,T,s,K=100,inn=100,call=True):
    delta_t = 1./252. 
    news=s[-1][np.where((s>=inn).any(axis=0))]
    
    c=np.zeros(s.shape[1])
    c[np.where((s>=inn).any(axis=0))]=option_help(news,K,call)
        
    return np.exp(-r*T*delta_t)*c
def get_options(inp):
    return [{'label': c, 'value': c} for c in inp]    


# ## stock

# ### stock info

# In[4]:


global ticker_list
ticker_list = ['AAPL', 'TSLA', 'MSFT', 'BAC', 'GS', 'AAL']

def generate_ticker_button(ticker_list):
    button_group = dbc.ButtonGroup([dbc.Button(i,id=i+'_btn',n_clicks=0) for i in ticker_list],id='ticker_group')
    return button_group

def load_data(ticker_list=ticker_list):
    
    global stock,prices,volume,dividend,returns,cul_returns,df
    
    myTickers = yf.Tickers(ticker_list)
    stock = myTickers.history(period="10y")

    prices = stock['Close'].copy()
    volume = stock['Volume'].copy()
    dividend = stock['Dividends'].copy()

    returns = (prices/prices.shift(1)).apply(np.log, axis = 1)*100
    cul_returns =  (1 + returns/100).cumprod()

    df = pd.DataFrame()
    names=ticker_list
    for name in names:
        df_tmp = prices[[name]].merge(volume[[name]], left_index = True, right_index = True).merge(
                                returns[[name]], left_index = True, right_index = True).merge(
                                dividend[[name]], left_index = True, right_index = True).dropna()
        df_tmp.columns = ['Price', 'Volume', 'Return','Dividend']
        df_tmp['name'] = name
        df = df.append(df_tmp)
    return stock,prices,volume,dividend,returns,cul_returns,df

stock,prices,volume,dividend,returns,cul_returns,df=load_data()
    
def get_data(freq,data):
    if freq == 'daily':
        return data
    elif freq == 'monthly':
        return data.resample('BM').apply(lambda x: x[-1])
    
    else:
        return data.resample('3M').mean()
    
def get_options(inp=ticker_list):
    return [{'label': c, 'value': c} for c in inp]  


# In[5]:


def plot1(data,name): #eg. data=get_prices(freq='daily')
    tickers = data.columns.to_list()
    fig = go.Figure()
    for tik in tickers:
        fig.add_trace(go.Scatter(y=data[tik].to_list(), x=data[tik].index.to_list(), visible = True))
    fig.update_layout(
        # this is a function taking multiple kwargs where complex args have to be passed as dictionaries
        title = {
            'text': name,
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 22}
        },
        paper_bgcolor = 'white',
        plot_bgcolor = 'white',
        autosize = False,
        height = 400,
        xaxis = {
            'title': 'Closing Date',
            'showline': True, 
            'linewidth': 1,
            'linecolor': 'black'
        },
        yaxis = {
            'showline': True, 
            'linewidth': 1,
            'linecolor': 'black'
        }
    )
    fig.for_each_trace(lambda t: t.update(name = tickers.pop(0)))
    fig.update_layout(
        xaxis=dict(
            rangeselector = dict(
                buttons = list([
                    dict(count=1,
                         label="1m",
                         step="month",
                        stepmode="backward"
                         ),
                    dict(count=6,
                         label="6m",
                         step="month",
                         # stepmode="backward"
                         ),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         # stepmode="todate"
                         ),
                    dict(count=1,
                         label="1y",
                         step="year",
                         # stepmode="backward"
                         ),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date",
        )
    )

    fig.layout.hovermode = 'x'

    #fig.show()
    return fig

def plot2(y='Return',label='Price',q=10,df=df):
    
    data = []
    names=list(df['name'].unique())
    for name in names:
        # We need to get the grouping by name
        this_name = df.loc[df['name'] == name].reset_index(drop=True)
        this_name['labels'] = pd.qcut(this_name[label], q)
        this_data = this_name[y].groupby(this_name['labels']).mean()
        data.append(
            go.Bar(
                name = name, 
                x = [str(round(x.left, 2)) + '-' + str(round(x.right, 2)) for x in this_data.index.categories],
                y = this_data.to_list()
            )
        )
        
    fig = go.Figure(data = data)

    fig.update_layout(
        barmode = 'group',
        title = 'Bar Chart of Equity Returns grouped by '+label,
        paper_bgcolor = 'white',
        plot_bgcolor = 'white',
        xaxis = dict(
            showline = True, 
            linewidth = 2, 
            linecolor = 'black'
        ),
        yaxis=dict(
            title = 'Stock '+y,
            titlefont_size = 16,
            tickfont_size = 14,
            gridcolor = '#dfe5ed'
        )
    )

    fig.layout.hovermode = 'x'
    
    return(fig)


def divided_plot(dividend=dividend):
    fig = go.Figure()

    d=dividend[dividend>0].dropna(how='all')
    names=ticker_list
    for name in names:
        size= d[name].fillna(0).to_list()
        fig.add_trace(go.Scatter(y = d[name].to_list() , x = d[name].index.to_list(), 
                             mode = 'markers', 
                                 marker=dict(
                                         size=size,
                                         sizemode='area',
                                         sizeref=2./(30.**2),
                                         #sizemin=4
                                        ),
                                 name=name,
                                 #legendwidth=1000
                                ))

       # fig.update_traces(selector = {'name': name}, visible = 'legendonly')

        fig.update_layout(
            # this is a function taking multiple kwargs where complex args have to be passed as dictionaries
            title = {
                'text': 'Stock Dividend of different companies',
                'y': 0.95,
                'x': 0.5,
                'font': {'size': 22}
            },
            paper_bgcolor = 'white',
            plot_bgcolor = 'white',
            autosize = False,
            height = 400,
            xaxis = {
                'title': 'Date',
                'showline': True, 
                'linewidth': 1,
                'linecolor': 'black',
            },
            yaxis = {
                'title': 'Dividend',
                'showline': True, 
                'linewidth': 1,
                'linecolor': 'black'
            }
        )

    #     fig.update_layout(
    #         xaxis=dict(
    #             rangeslider=dict(
    #                 visible=True
    #             ),
    #         )
    #     )
        fig.layout.hovermode = 'x'

    return fig


divid_fig3=divided_plot(dividend)


# In[6]:


def evo_bar_plot():
    P=prices.resample('Y').mean()
    P=pd.DataFrame(P.stack()).reset_index()
    P['year']=[(P['Date'][i]).date().year for i in range(len(P))]
    P=P.rename(columns={'level_1':'comp',0:'price'})

    fig = px.bar(P,
                 y="comp",
                 x='price',
                 animation_frame="year",
                 orientation='h',
                 range_x=[0, P.price.max()],
                 color='price',
                 #color_continuous_scale=px.colors.diverging.Temps
                 color_continuous_scale=px.colors.sequential.deep
                 )
    # improve aesthetics (size, grids etc.)
    fig.update_layout(width=1000,
                      height=800,
                      xaxis_showgrid=False,
                      yaxis_showgrid=False,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      title_text='Evolution of Stock Price in 10 years',
                      showlegend=False)
    fig.update_xaxes(title_text='Stock price')
    fig.update_yaxes(title_text='')
    return fig


# ### backtest

# In[7]:


def BackTest(ticker, strategy, fd, td):
    #get_ipython().run_line_magic('matplotlib', 'inline')
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000.0)
    stock1 = bt.feeds.PandasData(dataname = yf.download(ticker, fd, td, auto_adjust=True))
    cerebro.adddata(stock1)
    cerebro.addstrategy(strategy)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    cerebro.addanalyzer(bt.analyzers.Calmar, _name='_Calmar')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio')
    cerebro.addanalyzer(bt.analyzers.PeriodStats, _name='_PeriodStats')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='_SQN')
    cerebro.addanalyzer(bt.analyzers.VWR, _name='_VWR')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='_TradeAnalyzer')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')

    cerebro.broker.setcommission(commission=0.005) 
    results = cerebro.run()

    performance_dict=collections.OrderedDict()
    calmar_ratio = list(results[0].analyzers._Calmar.get_analysis().values())[-1]
    drawdown_info=results[0].analyzers._DrawDown.get_analysis()
    average_drawdown_len=drawdown_info['len']
    average_drawdown_rate=drawdown_info['drawdown']
    average_drawdown_money=drawdown_info['moneydown']
    max_drawdown_len=drawdown_info['max']['len']
    max_drawdown_rate=drawdown_info['max']['drawdown']
    max_drawdown_money=drawdown_info['max']['moneydown']
    sharpe_info=results[0].analyzers._SharpeRatio.get_analysis()
    if(sharpe_info['sharperatio'] == None):
        sharpe_ratio = 0
    else:
        sharpe_ratio=sharpe_info['sharperatio']
    PeriodStats_info=results[0].analyzers._PeriodStats.get_analysis()
    average_rate=PeriodStats_info['average']
    stddev_rate=PeriodStats_info['stddev']
    positive_year=PeriodStats_info['positive']
    negative_year=PeriodStats_info['negative']
    nochange_year=PeriodStats_info['nochange']
    best_year=PeriodStats_info['best']
    worst_year=PeriodStats_info['worst']
    SQN_info=results[0].analyzers._SQN.get_analysis()
    sqn_ratio=SQN_info['sqn']
    VWR_info=results[0].analyzers._VWR.get_analysis()
    vwr_ratio=VWR_info['vwr']
    annual_return = results[0].analyzers._AnnualReturn.get_analysis()

#     performance_dict['calmar_ratio']=calmar_ratio
#     performance_dict['average_drawdown_len']=average_drawdown_len
#     performance_dict['average_drawdown_rate']=average_drawdown_rate
#     performance_dict['average_drawdown_money']=average_drawdown_money
#     performance_dict['max_drawdown_len']=max_drawdown_len
    performance_dict['Max Drawdown Rate']=round(max_drawdown_rate,2)
#     performance_dict['max_drawdown_money']=max_drawdown_money
    performance_dict['Sharpe Ratio']=round(sharpe_ratio,2)
#     performance_dict['average_rate']=average_rate
    performance_dict['Standard Deviation']=round(stddev_rate,2)
#     performance_dict['positive_year']=positive_year
#     performance_dict['negative_year']=negative_year
#     performance_dict['nochange_year']=nochange_year
#     performance_dict['best_year']=best_year
#     performance_dict['worst_year']=worst_year
#     performance_dict['sqn_ratio']=sqn_ratio
#     performance_dict['vwr_ratio']=vwr_ratio
#     performance_dict['omega']=0
    performance_dict['Annual Return'] = round(np.array(list(annual_return.values())).mean()*100,2)

    
    trade_dict_1=collections.OrderedDict()
    trade_dict_2=collections.OrderedDict()
    trade_info=results[0].analyzers._TradeAnalyzer.get_analysis()
    total_trade_num=trade_info['total']['total']
#     total_trade_opened=trade_info['total']['open']
#     total_trade_closed=trade_info['total']['closed']
#     total_trade_len=trade_info['len']['total']
#     long_trade_len=trade_info['len']['long']['total']
#     short_trade_len=trade_info['len']['short']['total']

#     longest_win_num=trade_info['streak']['won']['longest']
#     longest_lost_num=trade_info['streak']['lost']['longest']
#     net_total_pnl=trade_info['pnl']['net']['total']
#     net_average_pnl=trade_info['pnl']['net']['average']
#     win_num=trade_info['won']['total']
#     win_total_pnl=trade_info['won']['pnl']['total']
#     win_average_pnl=trade_info['won']['pnl']['average']
#     win_max_pnl=trade_info['won']['pnl']['max']
#     lost_num=trade_info['lost']['total']
#     lost_total_pnl=trade_info['lost']['pnl']['total']
#     lost_average_pnl=trade_info['lost']['pnl']['average']
#     lost_max_pnl=trade_info['lost']['pnl']['max']

#     trade_dict_1['total_trade_num']=total_trade_num
#     trade_dict_1['total_trade_opened']=total_trade_opened
#     trade_dict_1['total_trade_closed']=total_trade_closed
#     trade_dict_1['total_trade_len']=total_trade_len
#     trade_dict_1['long_trade_len']=long_trade_len
#     trade_dict_1['short_trade_len']=short_trade_len
#     trade_dict_1['longest_win_num']=longest_win_num
#     trade_dict_1['longest_lost_num']=longest_lost_num
#     trade_dict_1['net_total_pnl']=net_total_pnl
#     trade_dict_1['net_average_pnl']=net_average_pnl
#     trade_dict_1['win_num']=win_num
#     trade_dict_1['win_total_pnl']=win_total_pnl
#     trade_dict_1['win_average_pnl']=win_average_pnl
#     trade_dict_1['win_max_pnl']=win_max_pnl
#     trade_dict_1['lost_num']=lost_num
#     trade_dict_1['lost_total_pnl']=lost_total_pnl
#     trade_dict_1['lost_average_pnl']=lost_average_pnl
#     trade_dict_1['lost_max_pnl']=lost_max_pnl


    long_num=trade_info['long']['total']
    long_win_num=trade_info['long']['won']
    long_lost_num=trade_info['long']['lost']
    long_total_pnl=trade_info['long']['pnl']['total']
    long_average_pnl=trade_info['long']['pnl']['average']
    long_win_total_pnl=trade_info['long']['pnl']['won']['total']
    long_win_max_pnl=trade_info['long']['pnl']['won']['max']
    long_lost_total_pnl=trade_info['long']['pnl']['lost']['total']
    long_lost_max_pnl=trade_info['long']['pnl']['lost']['max']

    short_num=trade_info['short']['total']
    short_win_num=trade_info['short']['won']
    short_lost_num=trade_info['short']['lost']
    short_total_pnl=trade_info['short']['pnl']['total']
    short_average_pnl=trade_info['short']['pnl']['average']
    short_win_total_pnl=trade_info['short']['pnl']['won']['total']
    short_win_max_pnl=trade_info['short']['pnl']['won']['max']
    short_lost_total_pnl=trade_info['short']['pnl']['lost']['total']
    short_lost_max_pnl=trade_info['short']['pnl']['lost']['max']


#     trade_dict_2['long_num']=long_num
#     trade_dict_2['long_win_num']=long_win_num
#     trade_dict_2['long_lost_num']=long_lost_num
#     trade_dict_2['long_total_pnl']=long_total_pnl
#     trade_dict_2['long_average_pnl']=long_average_pnl
#     trade_dict_2['long_win_total_pnl']=long_win_total_pnl
#     trade_dict_2['long_win_max_pnl']=long_win_max_pnl
#     trade_dict_2['long_lost_total_pnl']=long_lost_total_pnl
#     trade_dict_2['long_lost_max_pnl']=long_lost_max_pnl
#     trade_dict_2['short_num']=short_num
#     trade_dict_2['short_win_num']=short_win_num
#     trade_dict_2['short_lost_num']=short_lost_num
#     trade_dict_2['short_total_pnl']=short_total_pnl
#     trade_dict_2['short_average_pnl']=short_average_pnl
#     trade_dict_2['short_win_total_pnl']=short_win_total_pnl
#     trade_dict_2['short_win_max_pnl']=short_win_max_pnl
#     trade_dict_2['short_lost_total_pnl']=short_lost_total_pnl
#     trade_dict_2['short_lost_max_pnl']=short_lost_max_pnl
    performance_dict['Win Rate'] = round((long_win_num/long_num)*100,2)

    df01=pd.DataFrame([performance_dict]).T
    df01.reset_index(inplace = True)
    df01.columns=['Performance Indicators', 'Values']
    df01.set_index('Performance Indicators', inplace = True)
#     df02=pd.DataFrame([trade_dict_1]).T
#     df02.columns=['norm trade indicators']
#     df03=pd.DataFrame([trade_dict_2]).T
#     df03.columns=['long/short trade indicators']
#     df00['performance indicators']=df01.index
#     df00['performance values']=[round(float(i),4) for i in list(df01['performance indicators'])]
#     df00['norm trade indicators']=df02.index
#     df00['norm trade values']=[round(float(i),4) for i in list(df02['norm trade indicators'])]
#     df00['long/short trade indicators']=df03.index
#     df00['long/short trade values']=[round(float(i),4) for i in list(df03['long/short trade indicators'])]
    
    scheme = PlotScheme(decimal_places=5, max_legend_text_width=20)
    figs = cerebro.plot(BacktraderPlotly(show=False, scheme=scheme))
    fig0 = go.Figure()
    fig0.add_trace(figs[0][0]['data'][0].update({'line': {'color': '#219ebc'}}))
    fig0.add_trace(figs[0][0]['data'][1].update({'line': {'color': '#ffb703'}}))

    fig0.update_layout(
        # this is a function taking multiple kwargs where complex args have to be passed as dictionaries
        title = {
            'text': 'Broker & Value',
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 22,'color':'#277da1'}
        },
        paper_bgcolor = 'white',
        plot_bgcolor = 'white',
        autosize = True,
        #height = 400,
        xaxis = {
            'title': 'Closing Date',
            'title': {'font':{'size':15}},
            'tickfont':{'size':15},
            'showline': True, 
            'linewidth': 1,
            'linecolor': '#023047',
            'color':'#277da1',
        },
        yaxis = {
            'title': 'Price',
            'title': {'font':{'size':15}},
            'tickfont':{'size':15},
            'showline': True, 
            'linewidth': 1,
            'linecolor': '#023047',
            'color':'#277da1',
            #'size': 10,
        },

    )

    fig0.update_layout(
        xaxis=dict(
            rangeselector = dict(
                buttons = list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         # stepmode="backward"
                         ),
                    dict(count=6,
                         label="6m",
                         step="month",
                         # stepmode="backward"
                         ),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         # stepmode="todate"
                         ),
                    dict(count=1,
                         label="1y",
                         step="year",
                         # stepmode="backward"
                         ),
                    dict(step="all")
                ]),
                activecolor='#e76f51',
                bgcolor='#83c5be',
            ),
            rangeslider=dict(
                visible=True,
            ),
            type="date",
        ),
        hovermode = 'x'
    )
    fig1 = go.Figure()
    fig1.add_trace(figs[0][0]['data'][8].update({'name':'EMA(12)','line': {'color': '#bc6c25'}}))
    fig1.add_trace(figs[0][0]['data'][5].update({'name':'Candlestick of price','increasing': {'line': {'color': '#2a9d8f'}},'decreasing': {'line': {'color': '#e76f51'}}}))
    fig1.add_trace(figs[0][0]['data'][6].update({'name':'Buy','line': {'color': '#219ebc'},'marker': {'line': {'color': '#219ebc',}}}))
    fig1.add_trace(figs[0][0]['data'][7].update({'name':'Sell','line': {'color': '#ffb703'},'marker': {'line': {'color': '#ffb703',}}}))


    fig1.update_layout(
        # this is a function taking multiple kwargs where complex args have to be passed as dictionaries
        title = {
            'text': 'Buy & Sell',
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 22,'color':'#1d3557'}
        },
        paper_bgcolor = 'white',
        plot_bgcolor = 'white',
        autosize = True,
        #height = 400,
        xaxis = {
            'title': 'Closing Date',
            'showline': True, 
            'linewidth': 1,
            'linecolor': '#023047',
            'color':'#023047',
        },
        yaxis = {
            'title': 'Price',
            #'title': {'font':{'size':20}},
            #'tickfont':{'size':230},
            'showline': True, 
            'linewidth': 1,
            'linecolor': '#023047',
            'color':'#1d3557',
            #'size': 10,
        },

    )


    fig1.update_layout(
        xaxis=dict(
            rangeselector = dict(
                buttons = list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         # stepmode="backward"
                         ),
                    dict(count=6,
                         label="6m",
                         step="month",
                         # stepmode="backward"
                         ),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         # stepmode="todate"
                         ),
                    dict(count=1,
                         label="1y",
                         step="year",
                         # stepmode="backward"
                         ),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date",
        ),
        hovermode = 'x'
    )
    return fig0, fig1, df01
class MACD(bt.Strategy):
    params = (
        ('maperiod', 15),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    @staticmethod
    def percent(today, yesterday):
        return float(today - yesterday) / today

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.volume = self.datas[0].volume

        self.order = None
        self.buyprice = None
        self.buycomm = None

        me1 = bt.indicators.EMA(self.data, period=12)
        me2 = bt.indicators.EMA(self.data, period=26)
        self.macd = me1 - me2
        self.signal = bt.indicators.EMA(self.macd, period=9)

        bt.indicators.MACDHisto(self.data)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.bar_executed_close = self.dataclose[0]
            
            self.bar_executed = len(self)

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return


    def next(self):
        if self.order:
            return

        if not self.position:
            condition1 = self.macd[-1] - self.signal[-1]
            condition2 = self.macd[0] - self.signal[0]
            if condition1 < 0 and condition2 > 0:
                self.order = self.buy()

        else:
            condition = (self.dataclose[0] - self.bar_executed_close) / self.dataclose[0]
            if condition > 0.1 or condition < -0.1:
                self.order = self.sell()

                
class KDJ(bt.Strategy):
    def log(self, txt, dt=None):
        """ Logging function fot this strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print("%s, %s" % (dt.isoformat(), txt))

    @staticmethod
    def percent(today, yesterday):
        return float(today - yesterday) / today

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.volume = self.datas[0].volume

        self.order = None
        self.buyprice = None
        self.buycomm = None

        
        self.high_nine = bt.indicators.Highest(self.data.high, period=9)
        
        self.low_nine = bt.indicators.Lowest(self.data.low, period=9)
        
        self.rsv = 100 * bt.DivByZero(
            self.data_close - self.low_nine, self.high_nine - self.low_nine, zero=None
        )
        
        self.K = bt.indicators.EMA(self.rsv, period=3)
        
        self.D = bt.indicators.EMA(self.K, period=3)
        # J=3*K-2*D
        self.J = 3 * self.K - 2 * self.D

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.bar_executed_close = self.dataclose[0]
            
            self.bar_executed = len(self)

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return


    
    def next(self):
        if self.order:
            return

        condition1 = self.J[-1] - self.D[-1]
        condition2 = self.J[0] - self.D[0]
        if not self.position:
            
            if condition1 < 0 and condition2 > 0:
                self.order = self.buy()

        else:
            if condition1 > 0 or condition2 < 0:
                self.order = self.sell()

class KDJ_MACD(bt.Strategy):
    def log(self, txt, dt=None):
        """ Logging function fot this strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print("%s, %s" % (dt.isoformat(), txt))

    @staticmethod
    def percent(today, yesterday):
        return float(today - yesterday) / today

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.volume = self.datas[0].volume

        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.high_nine = bt.indicators.Highest(self.data.high, period=9)

        self.low_nine = bt.indicators.Lowest(self.data.low, period=9)
        
        self.rsv = 100 * bt.DivByZero(
            self.data_close - self.low_nine, self.high_nine - self.low_nine, zero=None
        )
        
        self.K = bt.indicators.EMA(self.rsv, period=3, plot=False)
        
        self.D = bt.indicators.EMA(self.K, period=3, plot=False)
        # J=3*K-2*D
        self.J = 3 * self.K - 2 * self.D

        
        me1 = bt.indicators.EMA(self.data, period=12)
        me2 = bt.indicators.EMA(self.data, period=26)
        self.macd = me1 - me2
        self.signal = bt.indicators.EMA(self.macd, period=9)
        bt.indicators.MACDHisto(self.data)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.bar_executed_close = self.dataclose[0]

            self.bar_executed = len(self)


        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return


    
    def next(self):
        if self.order:
            return

        if not self.position:
            condition1 = self.macd[-1] - self.signal[-1]
            condition2 = self.macd[0] - self.signal[0]
            if condition1 < 0 and condition2 > 0:
                self.order = self.buy()

        else:
            condition1 = self.J[-1] - self.D[-1]
            condition2 = self.J[0] - self.D[0]
            if condition1 > 0 or condition2 < 0:
                self.order = self.sell()


# ### MC

# In[8]:


def simulation_MC(df, days=50,trials=1000,observe_date=False,jump=False):
    
    if observe_date: 
        data=df.loc[observe_date[0]:observe_date[1]]
    else: #default using all history data
        data=df.copy()
    
    r_ind=[str(df.index[i].date()) for i in range(len(df))].index(observe_date[1])
    real_data=df.iloc[(r_ind):min(r_ind+days+1,len(df)),:]    
    
    
    log_returns = np.log(1 + data.pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u-0.5*var
    stdev = log_returns.std()
    Z = norm.ppf(np.random.rand(days, trials))
    daily_returns = np.exp(drift.values + stdev.values * Z)
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = data.iloc[-1]
    for t in range(1, days):
        price_paths[t] = price_paths[t-1]*daily_returns[t]
        
        
    pred_ind=list(real_data.index)+ [real_data.index[-1]+timedelta(days = i) for i in range(days-len(real_data))] 
    return price_paths,real_data,data,pred_ind
def MC_plot(ticker,n_days,n_paths,start_date,end_date):
    
    observe_date = [start_date,end_date]
    simulations,real_data,obs_data,pred_ind = simulation_MC(all_data[ticker], n_days, n_paths,observe_date)
    pred_sig=simulations.std(axis=1)
    pred=simulations.mean(axis=1)
    pred_upper=pred+1.96*pred_sig
    pred_lower=pred-1.96*pred_sig
    real_ind=[str(real_data.index[i].date()) for i in range(len(real_data))]
    obs_ind=[str(obs_data.index[i].date()) for i in range(len(obs_data))]
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(y=pred, x=pred_ind,name='Prediction data'))
    fig1.add_trace(go.Scatter(y=real_data['Price'].to_list(), x=real_ind,name='Real data', mode = 'lines'))
    fig1.add_trace(go.Scatter(y=obs_data['Price'].to_list(), x=obs_ind,name='Observe history data',mode = 'lines'))
    fig1.add_traces([go.Scatter(x = pred_ind, y = pred_upper,
                               mode = 'lines', line_color = 'rgba(0,0,0,0)',
                               showlegend = False),
                    go.Scatter(x = pred_ind, y = pred_lower,
                               mode = 'lines', line_color = 'rgba(0,0,0,0)',
                               name = '95% confidence interval',
                               fill='tonexty', fillcolor = 'rgba(255, 0, 0, 0.2)')])
    fig1.add_vline(x = observe_date[1], line_width=3, line_dash="dash", 
            line_color="green")
    fig1.update_layout(
            # this is a function taking multiple kwargs where complex args have to be passed as dictionaries
            title = {
                'text': 'Simulation Results: param:'+observe_date[0]+' to '+observe_date[1],
                'y': 0.95,
                'x': 0.5,
                'font': {'size': 22}
            },
            paper_bgcolor = 'white',
            plot_bgcolor = 'white',
            autosize = False,
            height = 400,
            xaxis = {
                'title': 'Closing Date',
                'showline': True, 
                'linewidth': 1,
                'linecolor': 'black'
            },
            yaxis = {
                'showline': True, 
                'linewidth': 1,
                'linecolor': 'black'
            }
        )
    fig1.update_layout(
            xaxis=dict(
                rangeselector = dict(
                    buttons = list([
                        dict(count=1,
                             label="1m",
                             step="month",
                            #stepmode="backward"
                             ),
                        dict(count=6,
                             label="6m",
                             step="month",
                             # stepmode="backward"
                             ),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             # stepmode="todate"
                             ),
                        dict(count=1,
                             label="1y",
                             step="year",
                             # stepmode="backward"
                             ),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date",
            )
        )
    initial_range = [pred_ind[0]-timedelta(days=len(pred_ind)), pred_ind[-1]]
    fig1['layout']['xaxis'].update(range=initial_range)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=[simulations[-1][:i+1].mean() for i in range(len(simulations[-1]))],mode='lines', 
                              x=list(range(n_paths)),name='Real data'))
    fig2.update_layout(
        # this is a function taking multiple kwargs where complex args have to be passed as dictionaries
        title = {
            'text': 'Convergence Figure',
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 22}
        },
        paper_bgcolor = 'white',
        plot_bgcolor = 'white',
        autosize = False,
        height = 400,
        xaxis = {
            'title': 'Closing Date',
            'showline': True, 
            'linewidth': 1,
            'linecolor': 'black'
        },
        yaxis = {
            'showline': True, 
            'linewidth': 1,
            'linecolor': 'black'
        }
    )
    return (fig1,fig2)


# In[9]:


names=ticker_list.copy()
tkrs=names.copy()
    
all_data = {}
for t in tkrs:
    all_data[t] = df[df['name'] == t][['Price']]
#BackTest
strategy_dict = {'MACD' : MACD, 'KDJ' : KDJ, 'KDJ_MACD' : KDJ_MACD}
#stg = 'KDJ_MACD'
#bt_fig0, bt_fig1, performance = BackTest(tkk, strategy_dict[stg], fd, td)
#generate history graph
index = ['time', 'ticker', 'strategy', 'from', 'to']
#cols = index + performance.index.tolist()
cols=index+['Max Drawdown Rate','Sharpe Ratio','Standard Deviation','Annual Return','Win Rate']
#history = pd.DataFrame(columns = cols)
#history.set_index(index, inplace = True)
#history.loc[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tkk, stg, fd, td] = performance['Values'].tolist() 
stg_opts = ['KDJ', 'MACD', 'KDJ_MACD']
bt_fig0 = go.Figure()
bt_fig1 = go.Figure()


# In[10]:


history = pd.DataFrame(columns = cols)
history.set_index(index, inplace = True)


# In[11]:


min_date = df.index[0].date()
max_date = df.index[-1].date()


# In[12]:


'''
option_type={
    'Vanilla':['European'],
    'Asian':['Average price'],
    'Binary':['Asset or nothing','Cash or nothing'],
    'Barrier':['Down and in','Down and out','Up and in','Up and out'],
    'Lookback':['fixed','floating']    
}
'''

# function mapping
option_func = {
    'European':euro_option,
    'Cash or nothing':cash_or_nothing,
    'Asset or nothing':asset_or_nothing,
    'Average price':average_price,
    'floating':floating_lookback,
    'fixed':fixed_lookback,
    'Down and out':down_and_out,
    'Down and in':down_and_in,
    'Up and in':up_and_in,
    'Up and out':up_and_out,
}

option_list = list(option_func.keys())


# In[13]:


def option_MC(s0,r,vol,T,K=100,trails=1000,inOut=100,option_type="European",isCall=True): 
    #pass
    s=generate_stock(s0,r,vol,T,trails=1000)
    if option_type in option_list[0:6]: #no need for inOut
        
        c = option_func[option_type](r,T,s,K,isCall)
    else:                               #need in/out
        c = option_func[option_type](r,T,s,K,inOut,isCall)
    
    cprice=c.mean().round(2)
    converge=[c[:i].mean() for i in range(1,len(c)+1)]
    return s,cprice,converge
    
def plot_option(s,converge):
    print(s.shape)
    fig1 = go.Figure()
    for path in range(s.shape[1]):
        fig1.add_trace(go.Scatter(y = s[:,path], x = list(range(s.shape[0])),showlegend=False))
        
    fig1.update_layout(
        title = {
        'text': 'Simulation of underlying',
        'y': 0.95,
        'x': 0.5,
        'font': {'size': 22}
    },
    paper_bgcolor = 'white',
    plot_bgcolor = 'white',
    autosize = False,
    height = 400,
    width=600,
    xaxis = {
        'title': 'Simulation Days',
        'showline': True, 
        'linewidth': 1,
        'linecolor': 'black'
    },
    yaxis = {
        'title': 'Underlying Price',
        'showline': True, 
        'linewidth': 1,
        'linecolor': 'black'
    }
    )
        
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y = converge, x = list(range(s.shape[1]))))
    fig2.update_layout(
        title = {
        'text': 'Convergence of Simulation',
        'y': 0.95,
        'x': 0.5,
        'font': {'size': 22}
    },
    paper_bgcolor = 'white',
    plot_bgcolor = 'white',
    autosize = False,
    height = 400,
    width=600,
    xaxis = {
        'title': 'Number of Simulation paths',
        'showline': True, 
        'linewidth': 1,
        'linecolor': 'black'
    },
    yaxis = {
        'title': 'Estimated Option Price',
        'showline': True, 
        'linewidth': 1,
        'linecolor': 'black'
    }
    )
    return fig1,fig2


# # layout

# In[14]:


def initial_fig(title,x,rang=True):
    ini_fig=go.Figure()
    ini_fig.update_layout(
    # this is a function taking multiple kwargs where complex args have to be passed as dictionaries
    title = {
        'text': title,
        'y': 0.95,
        'x': 0.5,
        'font': {'size': 22}
    },
    paper_bgcolor = 'white',
    plot_bgcolor = 'white',
    autosize = False,
    height = 400,
    xaxis = {
        'title': x,
        'showline': True, 
        'linewidth': 1,
        'linecolor': 'black'
    },
    yaxis = {
        'showline': True, 
        'linewidth': 1,
        'linecolor': 'black'
    }
    )
    if rang:
        ini_fig.update_layout(
            xaxis=dict(
                rangeselector = dict(
                    buttons = list([
                        dict(count=1,
                             label="1m",
                             step="month",
                            stepmode="backward"
                             ),
                        dict(count=6,
                             label="6m",
                             step="month",
                             # stepmode="backward"
                             ),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             # stepmode="todate"
                             ),
                        dict(count=1,
                             label="1y",
                             step="year",
                             # stepmode="backward"
                             ),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date",
            )
        )
    return ini_fig

ini_fig=initial_fig(None,None,False)


# ## option

# In[15]:


fig1=go.Figure
fig2=go.Figure()


# In[16]:


option_type={
    'Vanilla':['European'],
    'Asian':['Average price'],
    'Binary':['Asset or nothing','Cash or nothing'],
    'Barrier':['Down and in','Down and out','Up and in','Up and out'],
    'Lookback':['fixed','floating']    
}

def get_level1():
    return [{'label': c, 'value': c} for c in option_type.keys()]
def get_level2(level1):
    return [{'label': c, 'value': c} for c in option_type[level1]]


# In[17]:


call_put_button=dbc.Button(
            "Simulate",id="simulate",n_clicks=0,
            size="lg",
            className="me-2",
            style={'margin-top':'1em'}
        )


# In[18]:


#app = JupyterDash(__name__,external_stylesheets=[dbc.themes.YETI]) 

option_types = html.Div(
    [
        html.H2('Option Pricing with Monte Carlo Methods'),
       
        # There are hidden options here
        html.Br(),
        dbc.Row(
        [dbc.Col(
            [html.Label('Options type1:'),
                dcc.Dropdown(
                    options = get_level1(),
                    placeholder = 'Select...',
                    disabled = False,
                    searchable = True,
                    multi = False,
                    id = 'level1_dd'
                )]),
        html.Br(),
        dbc.Col(
        html.Div(
            [
                html.Label('Options type2:'),
                dcc.Dropdown(
                    options = [],
                    placeholder = 'No Level1 option selected',
                    disabled = True,
                    searchable = False,
                    multi = False,
                    id = 'level2_dd'
                )
            ], #style = {'width': '15em'}
        ),),
         
        dbc.Col(call_put_button ),
        html.Hr(),])
        # Let's add something else together!
        
    ], style = { 'padding': '1em'} #'border-style': 'solid',
)


# In[19]:


#app = JupyterDash(__name__,external_stylesheets=[dbc.themes.YETI]) 
input_groups = html.Div(
    dbc.Stack(
    [dbc.InputGroup(
       [ 
           dbc.InputGroupText("S0"),
            dbc.Input(id="option_s0", placeholder="Original underlying price",type="number",required=True),
       ]),
    dbc.InputGroup(
       [ 
           dbc.InputGroupText("T"),
           dbc.Input(id="option_T", placeholder="Simulation days",type="number",required=True),
       ]),
    dbc.InputGroup(
       [ 
           dbc.InputGroupText("Paths"),
             dbc.Input(id="option_trails", placeholder="Simulation paths",type="number",required=True),
       ]),
    dbc.InputGroup(
       [ 
           dbc.InputGroupText("r"),
            dbc.Input(id="option_r", placeholder="Risk-neutral rate (annual)",type="number",required=True),
       ]),
    dbc.InputGroup(
       [ 
           dbc.InputGroupText("vol"),
           dbc.Input(id="option_vol", placeholder="Volatility of underlying (annual)",type="number",required=True),
       ]),
    dbc.InputGroup(
       [ 
           dbc.InputGroupText("K"),
           dbc.Input(id="option_K", placeholder="Stike price",type="number"),
       ]),
    dbc.InputGroup(
       [ 
           dbc.InputGroupText("Q/IN/OUT"),
           dbc.Input(id="option_QIO", placeholder="Q/Knock-in/Knock-out Level",type="number"),
       ]),
    dbc.InputGroup(
        [
            dbc.RadioItems(
            options=[
                {"label": "Call", "value": True},
                {"label": "Put", "value": False},
                
            ],
            value=True,
            style={'size':30,'font':{'size':35}} ,           
            id="call or put",
        ),
        ]
    )
    ],gap=3)
    ,style = {'padding-up': '3em','padding-right': '2em','padding-left': '0.2em'}
     
 #,style = {'border-style': 'solid', 'padding': '5em'}
)



# app.layout = html.Div([input_groups])
# if __name__ == '__main__':
#     app.run_server(
#         debug=True,
#         port='8090',
#         #inline=True
#                   )


# In[20]:



op_card = html.Div(dbc.Card(
    [
       
            dbc.Tabs(
                [
                    dbc.Tab(label="Simulation", tab_id="op-tab-1"),
                    dbc.Tab(label="Convergence", tab_id="op-tab-2"),
                ],
                id="op-tabs",
                active_tab="op-tab-1",
            ),
        
        html.Div(dcc.Graph(figure=initial_fig("Simulation","Simulation Days",rang=False),id="op-content")),]
               
    
))



# In[21]:


#mc page
row = html.Div([
    dbc.Row([option_types,
    
    dbc.Row([
        dbc.Col(input_groups,width='auto'),
        dbc.Col(
        [
            dbc.Row(html.H2(id='cprice')),
            dbc.Row(op_card),
        ],width='auto')
    ],align="center"
    )
                 
]),]
)


# ## stock

# In[22]:


fre_price_tk=dcc.Dropdown(
                    options = get_options(ticker_list),
                    placeholder = 'Select...',
                    disabled = False,
                    searchable = True,
                    multi = True,
                    id = 'price tkr options'
                )
fre_price_card = html.Div(dbc.Card(
    [       
            dbc.Tabs(
                [
                    dbc.Tab(label="Daily", tab_id="fre-tab-1"),
                    dbc.Tab(label="Monthly", tab_id="fre-tab-2"),
                    dbc.Tab(label="Seasonly", tab_id="fre-tab-3"),
                ],
                id="fre-price-tabs",
                active_tab="fre-tab-1",
            ),
        
        #html.Div(id="fre_price_content"),
        dcc.Graph(id="fre_price_content"),
        ]                   
))

fre_re_tk=dcc.Dropdown(
                    options = get_options(ticker_list),
                    placeholder = 'Select...',
                    disabled = False,
                    searchable = True,
                    multi = True,
                    id = 'cum return tkr options'
                )
fre_re_card = html.Div(dbc.Card(
    [       
            dbc.Tabs(
                [
                    dbc.Tab(label="Daily", tab_id="re-tab-1"),
                    dbc.Tab(label="Monthly", tab_id="re-tab-2"),
                    dbc.Tab(label="Seasonly", tab_id="re-tab-3"),
                ],
                id="fre-re-tabs",
                active_tab="re-tab-1",
            ),
        
        #html.Div(id="fre_price_content"),
        dcc.Graph(id="cum_return_fig"),
        ]                   
))

bar_card = html.Div(dbc.Card(
    [       
            dbc.Tabs(
                [
                    dbc.Tab(label="Price", tab_id="bar-tab-1"),
                    dbc.Tab(label="Volume", tab_id="bar-tab-2"),
                ],
                id="bar-tabs",
                active_tab="bar-tab-1",
            ),
        
        #html.Div(id="fre_price_content"),
        dcc.Graph(id="bar_fig"),
        ]                   
))



# In[23]:


global pre_ticker
ticker_list = ['AAPL', 'TSLA', 'MSFT', 'BAC', 'GS', 'AAL']
pre_ticker=ticker_list.copy()

btn_g=generate_ticker_button(ticker_list)
row1=dbc.Row([
    dbc.Col([
    
        html.Label('Please input company  tickers'),
        html.Br(),
        dbc.Input(id="ticker_input", placeholder="Type something...", type="text"),
   ],width='auto'),
        
    dbc.Col([dbc.Button('+',id='plus_ticker_btn',n_clicks=0,style={'border-radius':'5px','height':50,'width':50,'font-size':25})]),
   
],align="end",)


# In[24]:


tab1_content = [html.Div([
                        dbc.Alert("Ticker name exisits or is invalid !",id="alert-auto",is_open=False,
            duration=2000,color='danger'),
                        row1,
                       html.Br(),
                       html.Div(btn_g,id='select_tick_btn'),
                       html.Br(),
                       fre_price_tk,
                       fre_price_card,                      
                       html.Br(),
                       fre_re_tk,
                       fre_re_card,
                       html.Br(),
                       bar_card,
                       html.Br(),
                        dcc.Graph(figure=divid_fig3,id='div_fig'),
                       dcc.Graph(id='evo_fig'),
                       
                      ])]

tab2_content = [
    html.Div(
    [
        html.H2('Back Test with Stocks'),

        html.Br(),
        html.Div(children=[
            dcc.Dropdown(
                options = get_options(stg_opts),
                placeholder = 'Select...',
                disabled = False,
                searchable = True,
                multi = False,
                id = 'strategy options',
                style={'width': 220}
            ),

            dcc.Dropdown(
                options = get_options(tkrs),
                placeholder = 'Select...',
                disabled = False,
                searchable = True,
                multi = False,
                id = 'ticker options', 
                style={'width': 220}
            )
        ], style={'display':'flex'}),
        
        dmc.DateRangePicker(
            id="date range picker2",
            label="Date Range",
            description="Choose BackTest period",
            minDate=min_date,
            maxDate=max_date,
            #value=[datetime.now().date(), datetime.now().date() + timedelta(days=5)],
            style={"width": 330},
        ),
        html.Div(children = [
            dbc.Button('Start backtest', id = 'backtest')
        ], style = {'margin' : '1em'}),
        html.Br(),
        
        html.Div(children=[
            dbc.Card(
            [
                dbc.Tabs(
                [
                    dbc.Tab(label = 'Total Value', tab_id = 'total_val'),
                    dbc.Tab(label = 'B/S Signals', tab_id = 'bs_signals')
                ],
                id = 'bt_tabs',
                active_tab = 'total_val',
                ),
                html.Div(id = 'bt_graphs'),
            ], style = {'width': '60%', 'height':'35%', 'margin': '1em'},
            ),
            html.Div(id='strategy performance')
            
        ], style={'display':'flex'}),
        html.Div(children=[
            html.Div(id='performance history')
        ], style = {'justify-content':'center'})
        
        
    ], style = {'margin': '1em', 'padding': '1em'}
)
]

tab3_content = [
    html.Div(
    [
        html.H2('Monte-Carlo Simulation'),
        html.Br(),
        html.Div(
            [
                dbc.Card(
                [
                    dcc.Dropdown(
                        options = get_options(tkrs),
                        placeholder = "Choose ticker",
                        disabled = False,
                        searchable = True,
                        multi = False,
                        id = "MC tkr options1"),
                    dmc.DateRangePicker(
                        id="date range picker",
                        label="Date Range",
                        description="Choose MC calculation period",
                        minDate=min_date+timedelta(days=30),
                        maxDate=max_date,
                        #value=[datetime.now().date(), datetime.now().date() + timedelta(days=5)],
                        style={"width": 330},
                    ),
                    html.Div([
                        html.Div([daq.NumericInput(
                            min=1,
                            max=200,
                            value=50,
                            label='Simultaion days',
                            labelPosition='bottom',
                            id = "MC simulation days"
                        )], style = {'margin':'1em'}),
                        html.Div([
                            daq.NumericInput(
                                min=50,
                                max=1000,
                                value=100,
                                label='Number of paths',
                                labelPosition='bottom',
                                id = "MC path nums"
                        )  
                        ], style = {'margin':'1em'})                          
                    ], style = {'display':'flex', 'margin':'auto'}),

                    html.Div([
                            dbc.Button('Simulate', id = 'simulate button', n_clicks = 0, style = {'width':110}),
                    ], style = {'margin':'auto'})
                    
                ],style = {'margin':'auto'}
                ) 
                
            ],style={'display':'flex'}
               
        ),
        html.Div(children=[
            dbc.Card(
            [
                dcc.Tabs(
                [
                    dcc.Tab(label = 'Monte Carlo', id = 'tab_MC', children = [dcc.Graph(id='MC fig1')]),
                    dbc.Tab(label = 'Convergence', id = 'tab_Conv', children = [dcc.Graph(id='Converge fig')])
                ],
                id = 'MC_tabs'
                )
            ], style = {'width': '80%', 'height':'35%', 'margin': 'auto'},
            )
        ])
    ]
)
]



tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(label="Stock Info", tab_id="tab-1",tab_style={"marginLeft": "auto"}),
                dbc.Tab(label="Back Test", tab_id="tab-2"),
                dbc.Tab(label="Monte Carlo", tab_id="tab-3"),
            ],
            id="up_tabs",
            active_tab="tab-1",
        ),
        html.Div(id="content_tab"),
    ]
)


# ## Home

# In[25]:


carousel = dbc.Carousel(
    items=[
        {"key": "1", "src": "assets/pic2.png"},
        {"key": "2", "src": "assets/pic3.png"},
        {"key": "3", "src": "assets/pic1.png"},
        {"key": "4", "src": "assets/pic4.png"},
    ],
    controls=True,
    indicators=True,
    interval=2000,
    ride="carousel",
)



card1 = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Nanyu Jiang", className="card-title",style={'color':'#0077b6'}),
            html.H6("nj2216", className="card-subtitle"),
            #html.P("N13688103",className="card-text",),
        ]
    ),
    style={"width": "18rem"},
)

card2 = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Zeyu Guo", className="card-title",style={'color':'#0077b6'}),
            html.H6("zg2175", className="card-subtitle"),
            #html.P("N13688103",className="card-text",),
        ]
    ),
    style={"width": "18rem"},
)

card3 = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Shunyao Xu", className="card-title",style={'color':'#0077b6'}),
            html.H6("sx2233", className="card-subtitle"),
            #html.P("N13688103",className="card-text",),
        ]
    ),
    style={"width": "18rem"},
)

card=dbc.Row(
            [
                dbc.Col(card1,md=4),
                dbc.Col(card2,md=4),
                dbc.Col(card3,md=4),
            ])


# # menu

# In[26]:


# app = JupyterDash(__name__,
#                  prevent_initial_callbacks = True,
#                  suppress_callback_exceptions=False)     # in a python py file this should be app = dash.Dash()
#app = dash.Dash(__name__)



#app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}


CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

image_url = "https://lh3.googleusercontent.com/Y5Bzn4AmS6Xt1eOcoP5U7E8bEI9aNZnA88uYfpkx089udtWxyu2RCc9WKxdC6HoBrQMg7XLVelJ4clH2sin1L3ELtqQm-93kdgC_7RQJ"

sidebar = html.Div(
    [
        #html.H2("Sidebar", className="display-4"),
        #html.Img(src='nyu.jpg', style = {'height': '100px', 'width': '200px'}),className="rounded-circle"
        html.Img(src=image_url, style = {'height': '160px', 'width': '220px'}),
        html.Hr(),
        html.P(
            "FRE-GY.6191 Visualization Lab", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Stock", href="/page-1", active="exact"),
                dbc.NavLink("Option", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)



menu1=html.Div(
            [html.Div([html.H1("Welcome!", className="card-text"),
            html.H4("This is our presentation for visualization lab !"),]),
            html.Br(),
            card,
            html.Br(),
            dbc.Col(carousel,width={"size": 6, "offset": 3})]    
    )

menu2=html.Div(
        [html.H4("Here you can play around with stocks"),tabs] 
    )

menu3=html.Div(
        [row] 
    )


content = html.Div(id="page-content", style=CONTENT_STYLE)

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Page 0", href="#")),
    ],
    brand="NavbarSimple",
    brand_href="#",
    color="primary",
    dark=True,
)


# # APP callback

# In[27]:


app = JupyterDash(__name__,external_stylesheets=[dbc.themes.YETI],
                 prevent_initial_callbacks = False,
                suppress_callback_exceptions=True
                 ) 

# app = dash.Dash(__name__,external_stylesheets=[dbc.themes.YETI],prevent_initial_callbacks = False,
#                  suppress_callback_exceptions=True)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


# In[28]:


# The memoization method
m = {}
def memoize_MC(f):
    def wrapper(*args):             #this could be done with *args and **kwargs to handle anything always
        global m

        new_key = list(args) # convert args to list
        dates = args[2] #find the dates argument
        new_key[2] = tuple(dates) #convert dates to tuple
        args_key = tuple(new_key[1:])
        
        # check if the arguments have been used before, if not, store them to the dictionary 
        if args_key not in m.keys():
            # this stores the results of the function in the args key

            m[args_key] = f(*args)
            #print('the results are ', m[args])
        # return the results
      
        return(m[args_key])
    
    # always return the inner function
    return(wrapper)


# In[29]:


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname): # 3 main pages
    if pathname == "/":
        return menu1
    elif pathname == "/page-1":
        return menu2
    elif pathname == "/page-2":
        return menu3
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


@app.callback(Output("content_tab", "children"), [Input("up_tabs", "active_tab")])
def switch_tab(at): #stock 3tabs
    if at == "tab-1":
        return tab1_content
    elif at == "tab-2":
        return tab2_content
    elif at == "tab-3":
        return tab3_content
    return html.P("This shouldn't ever be displayed...")



# ## options

# In[30]:


@app.callback(
    [
        Output('level2_dd', 'options'),
        Output('level2_dd', 'disabled'),
    ],
    Input('level1_dd', 'value'),
    State('level1_dd', 'disabled')
)
def choose_options(level1,status): #option chooser
    if level1:
        return(get_level2(level1), False)
    else:
        return([], True)

opt_results={}

@app.callback( 
    
    [
        Output('cprice','children'),
        Output("op-content", "figure"),
    ],
    
    Input('simulate','n_clicks'),
    Input("op-tabs", "active_tab"),
    [ 
        State('level2_dd','value'),
        State('option_s0','value'),
        State('option_r','value'),
        State('option_vol','value'),
        State('option_T','value'),
        State('option_K','value'),
        State('option_trails','value'),
        State('option_QIO','value'),
        State('call or put','value')
    ]            
)
def update_option(clicks,at,level2,s0,r,vol,T,K,trails,QIO,callorPut):
    if clicks ==0:
        return ["The _________ option price is ____."],ini_fig
    global fig_11,fig_22,opt_results
    
    current_params = tuple([clicks,level2,s0,r,vol,T,K,trails,QIO,callorPut])
    
    if current_params not in opt_results.keys(): #new parameters
        ss,cprice,converge = option_MC(s0,r,vol,T,K,trails,QIO,level2,callorPut)
        fig_11,fig_22 = plot_option(ss,converge)
        opt_results[current_params] = [fig_11,fig_22,cprice]
    else:
        cprice = opt_results[current_params][2]
        fig_11 = opt_results[current_params][0]
        fig_22 = opt_results[current_params][1]
        
    if callorPut:
        opt = 'call'
    else: 
        opt = 'put'
    price_str = f"The {opt} option price is {cprice}."

    if at == "op-tab-1":
        return [price_str],fig_11
    elif at == "op-tab-2":
        return [price_str],fig_22
    
'''
@app.callback(
    Output("op-content", "figure"), [Input("op-tabs", "active_tab")]
)
def op_tab_content(at): #option pic
#     op_fig1=[dcc.Graph(figure=fig1
#                    )]
#     op_fig2=[dcc.Graph(figure=fig2
#                    )]
    if at == "op-tab-1":
        return fig1#op_fig1
    elif at == "op-tab-2":
        return fig2#op_fig2
    return initial
'''


# ## stock MC

# In[31]:



@app.callback(
    [Output("MC fig1","figure"),
    Output("Converge fig","figure")],
    [Input("simulate button","n_clicks"),
    State("MC tkr options1","value"),
    State("date range picker","value"),
    State("MC simulation days","value"),
    State("MC path nums","value")]  
)
@memoize_MC
def update_MC(n_clicks,tkr,date_range,sim_days,path_nums):
    
    start_date, end_date = date_range
    MC_fig,Conv_fig = MC_plot(tkr,sim_days,path_nums,start_date,end_date)
    return (MC_fig,Conv_fig)


# ## Stock info

# In[32]:


@app.callback(
    [Output("select_tick_btn", "children"),Output("div_fig", "figure"),
    Output("price tkr options","options"),Output("cum return tkr options","options"),
     Output("evo_fig", "figure"),Output("alert-auto","is_open")
    ],
    [Input("plus_ticker_btn", "n_clicks"),State("ticker_input", "value")]
)
def ticker_pluc_button_click(n,tick):
    
    alert=False
    if tick is not None :
        tick=tick.upper()
    if tick in ticker_list:
        alert=True
    else:
        if tick is not None :
            myTickers = yf.Ticker(tick)
            stocks = myTickers.history(period="1mo")
            if stocks.empty: 
                alert=True
            else:
                ticker_list.append(tick)
                pre_ticker=ticker_list.copy()
            
    load_data(ticker_list)
    
    btn_g=generate_ticker_button(ticker_list)
    divid_fig=divided_plot(dividend)
    
    new_op=get_options(ticker_list)
    evo_fig=evo_bar_plot()
        
    return btn_g,divid_fig,new_op,new_op,evo_fig,alert#,fig2


@app.callback(
    Output("fre_price_content", "figure"), 
    [Input('price tkr options', 'value'),Input("fre-price-tabs", "active_tab")]
)
def op_tab_content(tkrs,at):
    if tkrs is not None:
        
        if at == "fre-tab-1":
            freq='daily'
        elif at == "fre-tab-2":
            freq='monthly'
        elif at == "fre-tab-3":
            freq='seasonly'
        price_data = get_data(freq=freq,data=prices[tkrs])
        price_fig = plot1(price_data,name='Stock Price')   
        return price_fig
    ini_fig=initial_fig('Stock Price','Closing Date')
    return ini_fig

@app.callback(
    Output("cum_return_fig", "figure"), 
    [Input('cum return tkr options', 'value'),Input("fre-re-tabs", "active_tab")]
)
def update_return(tkrs,at):
    if tkrs is not None:
        if at == "re-tab-1":
            freq='daily'
        elif at == "re-tab-2":
            freq='monthly'
        elif at == "re-tab-3":
            freq='seasonly'
        price_data = get_data(freq=freq,data=cul_returns[tkrs])
        price_fig = plot1(price_data,name='Culmulative Return')    #######need to correct
        #price_fig.for_each_trace(lambda t: t.update(visible = False))
    #     for opt in tkrs:
    #         price_fig.update_traces(selector = {'name': opt}, visible = True)
        return price_fig
    ini_fig=initial_fig('Culmulative Return','Closing Date')
    return ini_fig


@app.callback(
    Output("bar_fig", "figure"), 
    [Input("bar-tabs", "active_tab"),Input("plus_ticker_btn", "n_clicks")]
)
def bar_tabs(at,n):
    if at == "bar-tab-1":
        price_r_fig2=plot2(label='Price',df=df)
        return price_r_fig2
    elif at == "bar-tab-2":
        volum_r_fig2=plot2(label='Volume',df=df)
        return volum_r_fig2


# ## Backtest

# In[33]:


@app.callback(
    Output('bt_graphs', 'children'), 
    Output('strategy performance','children'),
    Output('performance history','children'),    
    [
        Input('backtest','n_clicks'),
        Input('bt_tabs', 'active_tab')],
    [
        State('ticker options','value'),
        State('strategy options','value'),
        State('date range picker2','value')
    ]
)
def bt_tab_content(n_clicks,at,tkk,stg,date_range):
    global history
    fd, td = date_range
    bt_fig0, bt_fig1, performance = BackTest(tkk, strategy_dict[stg], fd, td)
    performance_table = dbc.Table.from_dataframe(performance.reset_index(), striped=True, bordered=True, hover=True, index=False, 
                        style = {'width':350, 'height':250, 'margin':'1em'})
    if history.empty == False: #history has value
        if (list((history.index)[-1])[1:]) == [tkk,stg,str(fd),str(td)]:
            pass
        else:
            history.loc[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tkk, stg, fd, td] = performance['Values'].tolist() 
    
    else: #empty history
        history.loc[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tkk, stg, fd, td] = performance['Values'].tolist() 
    
    history_table = dbc.Table.from_dataframe(history.reset_index(), dark = True, striped=True, bordered=True, hover=True, index=False)
    if at == 'total_val':
        return dcc.Graph(figure=bt_fig0,id='fig0'),performance_table,history_table
    elif at == 'bs_signals':
        return dcc.Graph(figure=bt_fig1,id='fig1') ,performance_table,history_table 
    


# In[34]:


if __name__ == '__main__':
    app.run_server(
        debug=False,
        port='8288',
        #inline=True
                  )
    
#app.run_server(
    #mode = 'inline'
#)


# In[ ]:





# In[ ]:




