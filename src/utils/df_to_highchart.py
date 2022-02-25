# -*- coding: utf-8 -*-

import pandas
import copy
from highcharts import Highchart # !pip install python-highcharts
from IPython.display import display


#_colors = ['#7cb5ec', '#434348', '#90ed7d', '#f7a35c', '#8085e9', '#f15c80', '#e4d354', '#8085e8', '#8d4653', '#91e8e1'] #默认配色
#_colors = ['#d3d3d3','#60acfc' ,'#32d3eb','#5bc49f','#feb64d','#ff7c7c','#9287e7'] #彩虹色

_pd2hc_kind = {
    "bar": "column",
    "barh": "bar",
    "area": "area",
    "line": "line",
    "pie": "pie",
    "waterfall": "waterfall"
}


def pd2hc_kind(kind):
    if kind not in _pd2hc_kind:
        raise ValueError("%(kind)s plots are not yet supported" % locals())
    return _pd2hc_kind[kind]

_pd2hc_linestyle = {
    "-": "Solid",
    "--": "Dash",
    "-.": "DashDot",
    ":": "Dot"
}


def pd2hc_linestyle(linestyle):
    if linestyle not in _pd2hc_linestyle:
        raise ValueError("%(linestyle)s linestyles are not yet supported" % locals())
    return _pd2hc_linestyle[linestyle]


def json_encode(obj):
    return pandas.io.json.dumps(obj)


class Highchart_df(Highchart):
    def sethtml(self,_htmlcontent_):
        self._htmlcontent = _htmlcontent_
    
    def buildhtml(self):
        """build the HTML page
        create the htmlheader with css / js
        create html page
        """
#         self.buildcontent()
#         self.buildhtmlheader()
#         self.content = self._htmlcontent.decode('utf-8') # need to ensure unicode
#         self._htmlcontent = self.template_page_highcharts.render(chart=self)
        return self._htmlcontent

def serialize(df, output_type="javascript", chart_type="default", *args, **kwargs):
    """
    总结by jjh@201903,未知作用的欢迎添加
    
    序列化dataframe，转成hightchart可以展示的
    df,要序列化的dataframe
    output_type,输出的数据类型dict/json/javascript,默认为javascript
    
    colors,list like ['#d3d3d3','#60acfc' ,'#32d3eb','#5bc49f','#feb64d','#ff7c7c','#9287e7'],图表的样色
    
    legend, bool ,是否展示图例，默认为True
    
    title,string,图表标题

    x,作为X轴的column，默认不填，则用index作为X轴

    first_y_type,string,#主坐标的类型 ['column', #柱状图 
                                      'bar',
                                      'area',
                                      'line', #线
                                      'pie', #饼]

    secondary_y ,list, #次坐标的columns
    secondary_y_type,string ,次坐标的类型 like first_y_type
    tooltip,string ,数据标签展示的形式，这里是html
    secondary_y_tooltip,次轴数据标签
    zoom ,string in ('x','y','xy') #要缩放的轴
    visible,list,选中的columns，默认是全部
    not_visible,list,显示为不选中的columns，默认为[]

    
    """
    def serialize_chart(df, output, *args, **kwargs):
        output["chart"] = {}
        if 'render_to' in kwargs:
            output['chart']['renderTo'] = kwargs['render_to']
        if "figsize" in kwargs:
            output["chart"]["width"] = kwargs["figsize"][0]
            output["chart"]["height"] = kwargs["figsize"][1]
        if "kind" in kwargs:
            output["chart"]["type"] = pd2hc_kind(kwargs["kind"])
        if kwargs.get('polar'):
            output['chart']['polar'] = True

    def serialize_colors(df, output, *args, **kwargs):
        if 'colors' in kwargs:
            _colors_ = kwargs['colors']
            for i in range(len(output['series'])):
                output['series'][i]['color'] = _colors_[(i%len(_colors_))]
        else:
            pass

    def serialize_credits(df, output, *args, **kwargs):
        if 'credits' in kwargs:
            output['credits'] = { 'text':'{}'.format(kwargs['credits']['text']),'href':'{}'.format(kwargs['credits']['href'])}
        else:           
            output['credits'] = { 'text':'','href':''}

    def serialize_data(df, output, *args, **kwargs):
        pass

    def serialize_drilldown(df, output, *args, **kwargs):
        pass

    def serialize_exporting(df, output, *args, **kwargs):
        pass

    def serialize_labels(df, output, *args, **kwargs):
        pass

    def serialize_legend(df, output, *args, **kwargs):
        output["legend"] = {
            "enabled": kwargs.get("legend", True)
        }

    def serialize_loading(df, output, *args, **kwargs):
        pass

    def serialize_navigation(df, output, *args, **kwargs):
        pass

    def serialize_noData(df, output, *args, **kwargs):
        pass

    def serialize_pane(df, output, *args, **kwargs):
        pass

    def serialize_plotOptions(df, output, *args, **kwargs):
        if kwargs.get('plot_options'):
            output['plotOptions'] = kwargs['plot_options']
        else:
            pass



    def serialize_series(df, output, *args, **kwargs):
        def is_secondary(c, **kwargs):
            return c in kwargs.get("secondary_y", [])
        def not_visible(v, **kwargs):
            return v in kwargs.get("not_visible", [])

        if kwargs.get('sort_columns'):
            df = df.sort_index()
        series = df.to_dict('series')
        output["series"] = []
        for cn in df.columns:
            for name, data in series.items():
                if cn == name:
                    if df[name].dtype.kind in "biufc":
                        sec = is_secondary(name, **kwargs)
                        nvs = not_visible(name, **kwargs)
                        d = {
                            "name": name, #if not sec or not kwargs.get("mark_right", True) else name + " (right)",
                            "yAxis": int(sec),
                            "data": list(zip(df.index, data.values.tolist())),
                            "visible" : True if not nvs else False
                            # "type": kwargs['first_y_type'] if not sec else kwargs['secondary_y_type']
                        }
                        if kwargs.get('first_y_type'):
                            d['type'] = kwargs['first_y_type'] if not sec else kwargs['secondary_y_type']

                        if kwargs.get('secondary_y_tooltip') and sec:
                            d['tooltip'] = kwargs['secondary_y_tooltip']

                        if kwargs.get('polar'):
                            d['data'] = [v for k, v in d['data']]
                        if kwargs.get("kind") == "area" and kwargs.get("stacked", True):
                            d["stacking"] = 'normal'
                        if kwargs.get("style"):
                            d["dashStyle"] = pd2hc_linestyle(kwargs["style"].get(name, "-"))
                        output["series"].append(d)
                else: 
                    pass
        # output['series'].sort(key=lambda s: s['name'])

    def serialize_subtitle(df, output, *args, **kwargs):
        pass

    def serialize_title(df, output, *args, **kwargs):
        if "title" in kwargs:
            output["title"] = {"text": kwargs["title"]}

    def serialize_tooltip(df, output, *args, **kwargs):
        if 'tooltip' in kwargs:
            output['tooltip'] = kwargs['tooltip']

    def serialize_xAxis(df, output, *args, **kwargs):
        output["xAxis"] = {}
        if df.index.name:
            output["xAxis"]["title"] = {"text": df.index.name}
        if df.index.dtype.kind in "M":
            output["xAxis"]["type"] = "datetime"
        if df.index.dtype.kind == 'O':
            output['xAxis']['categories'] = sorted(list(df.index)) if kwargs.get('sort_columns') else list(df.index)
        if kwargs.get("grid"):
            output["xAxis"]["gridLineWidth"] = 1
            output["xAxis"]["gridLineDashStyle"] = "Dot"
        if kwargs.get("loglog") or kwargs.get("logx"):
            output["xAxis"]["type"] = 'logarithmic'
        if "xlim" in kwargs:
            output["xAxis"]["min"] = kwargs["xlim"][0]
            output["xAxis"]["max"] = kwargs["xlim"][1]
        if "rot" in kwargs:
            output["xAxis"]["labels"] = {"rotation": kwargs["rot"]}
        # if "unit" in kwargs:
        #     # print(kwargs["unit"])
        #     output["xAxis"]["labels"] = {"formatter": "function () {return this.value + '%s';}"%kwargs["unit"]}
        #     print(output["xAxis"]["labels"])

        if "fontsize" in kwargs:
            output["xAxis"].setdefault("labels", {})["style"] = {"fontSize": kwargs["fontsize"]}
        if "xticks" in kwargs:
            output["xAxis"]["tickPositions"] = kwargs["xticks"]

    def serialize_yAxis(df, output, *args, **kwargs):
        yAxis = {}
        if kwargs.get("grid"):
            yAxis["gridLineWidth"] = 1
            yAxis["gridLineDashStyle"] = "Dot"
        if kwargs.get("loglog") or kwargs.get("logy"):
            yAxis["type"] = 'logarithmic'
        if "ylim" in kwargs:
            yAxis["min"] = kwargs["ylim"][0]
            yAxis["max"] = kwargs["ylim"][1]
        if "rot" in kwargs:
            yAxis["labels"] = {"rotation": kwargs["rot"]}

        if "yformat" in kwargs:
            yAxis.setdefault("labels", {})['format'] = "{value} " + kwargs['yformat']

        if "fontsize" in kwargs:
            yAxis.setdefault("labels", {})["style"] = {"fontSize": kwargs["fontsize"]}
        if "yticks" in kwargs:
            yAxis["tickPositions"] = kwargs["yticks"]

        if "ytitle" in kwargs:
            yAxis["title"] = {"text": kwargs['ytitle']}
        else:
            yAxis["title"] = {"enabled": False}

        output["yAxis"] = [yAxis]
        if kwargs.get("secondary_y"):
            yAxis2 = copy.deepcopy(yAxis)
            if "ytitle2" in kwargs:
                yAxis2["title"]['text'] = kwargs['ytitle2']
            else:
                yAxis2["title"]['enabled'] = False
            if "ylim2" in kwargs:
                yAxis["min"] = kwargs["ylim2"][0]
                yAxis["max"] = kwargs["ylim2"][1]
                
            else:
                # yAxis2["title"]['enabled'] = False   
                pass
            if "syformat" in kwargs:
                yAxis2.setdefault("labels", {})['format'] = "{value} " + kwargs['syformat']

            yAxis2["opposite"] = True
            output["yAxis"].append(yAxis2)

    def serialize_zoom(df, output, *args, **kwargs):
        if "zoom" in kwargs:
            if kwargs["zoom"] not in ("x", "y", "xy"):
                raise ValueError("zoom must be in ('x', 'y', 'xy')")
            output["chart"]["zoomType"] = kwargs["zoom"]
        output["chart"]['resetZoomButton'] = {'position':{'align': 'left', 'y': -45}}

    def add_show_hide_all(json_output,*args,**kwargs):
        if not kwargs.get("add_hide_all",True)==False:
            json_output = json_output[:-1]+""",exporting:{
        buttons: {
            customButton: {
                text: 'Hide/Show All',
                onclick:
                    function () { var series = this.series[0]; if (series.visible) {for (var i = 0; i < this.series.length; i++) {this.series[i].hide();}} else {for (var i = 0; i < this.series.length; i++) {this.series[i].show();}}}
                }}

            }}"""
        return json_output

    output = {}
    df_copy = copy.deepcopy(df)
    if "x" in kwargs:
        df_copy.index = df_copy.pop(kwargs["x"])
    if kwargs.get("use_index", True) is False:
        df_copy = df_copy.reset_index()
    if "y" in kwargs:
        df_copy = pandas.DataFrame(df_copy, columns=kwargs["y"])
    serialize_chart(df_copy, output, *args, **kwargs)
    serialize_credits(df_copy, output, *args, **kwargs)
    serialize_data(df_copy, output, *args, **kwargs)
    serialize_drilldown(df_copy, output, *args, **kwargs)
    serialize_exporting(df_copy, output, *args, **kwargs)
    serialize_labels(df_copy, output, *args, **kwargs)
    serialize_legend(df_copy, output, *args, **kwargs)
    serialize_loading(df_copy, output, *args, **kwargs)
    serialize_navigation(df_copy, output, *args, **kwargs)
    serialize_noData(df_copy, output, *args, **kwargs)
    serialize_pane(df_copy, output, *args, **kwargs)
    serialize_plotOptions(df_copy, output, *args, **kwargs)
    serialize_series(df_copy, output, *args, **kwargs)
    serialize_subtitle(df_copy, output, *args, **kwargs)
    serialize_title(df_copy, output, *args, **kwargs)
    serialize_tooltip(df_copy, output, *args, **kwargs)
    serialize_xAxis(df_copy, output, *args, **kwargs)
    serialize_yAxis(df_copy, output, *args, **kwargs)
    serialize_zoom(df_copy, output, *args, **kwargs)
    serialize_colors(df_copy, output, *args, **kwargs)

    if output_type == "dict":
        return output
    if output_type == "json":
        return add_show_hide_all(json_encode(output),*args,**kwargs)
    if chart_type == "stock":
        return "new Highcharts.StockChart(%s);" % add_show_hide_all(json_encode(output),*args,**kwargs)
    if output_type == 'notebook':
        display_x_charts(add_show_hide_all(json_encode(output),*args,**kwargs))
    else:
        # raise ValueError("""不明输出类型,输出类型为["dict","json","stock","javascript","notebook"] 之一""")
        return "new Highcharts.Chart(%s);" % add_show_hide_all(json_encode(output),*args,**kwargs)

def get_template(num):
    notebook_template = """<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <link href="https://www.highcharts.com/highslide/highslide.css" rel="stylesheet" />
        <script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>
        <script type="text/javascript" src="https://code.highcharts.com/6/highcharts.js"></script>
        <script type="text/javascript" src="https://code.highcharts.com/6/highcharts-more.js"></script>
        <script type="text/javascript" src="https://code.highcharts.com/6/modules/heatmap.js"></script>
        <script type="text/javascript" src="https://code.highcharts.com/6/modules/exporting.js"></script>
    </head>

    <body style="margin:0;padding:0">
    {containers}
    <script>
    {scripts}
    </script>          
    </body>
</html>"""
    width = 100//num
    height = 500
    containers = ""
    scripts = ""
    for i in range(num):
        containers += '<div id="container{idx}" style="width:{width}%; float:left">    </div> \n'.format(idx=i+1,width = width,height=height)
        scripts += "$('#container{idx}').highcharts({{container{idx}}});".format(idx = i+1)
    return notebook_template.format(containers=containers,scripts=scripts)

def display_x_charts(json_charts,axis = 0):
    if isinstance(json_charts,list):
        dict_charts = {"container{}".format(i+1):chart for i,chart in enumerate(json_charts) }
    elif isinstance(json_charts,dict):
        dict_charts = {"container{}".format(i+1):json_charts[chart] for i,chart in enumerate(json_charts) }
    elif isinstance(json_charts,str):
        dict_charts = {"container1":json_charts}
    num = len(dict_charts)
    if axis == 1:
        notebook_template = get_template(num)     
        H = Highchart_df()
        html = notebook_template.format(**dict_charts)
        H.sethtml(html)
        display(H)
    else:
        notebook_template = get_template(1) 
        for chart in dict_charts.values():
            H = Highchart_df()
            html = notebook_template.format(container1 = chart)
            H.sethtml(html)
            display(H)   
            
table_string_cross = """{{chart: {{
                width:{width},
                height:{height},
                type: 'heatmap',
                marginTop: 100,
                marginBottom: 20,
                plotBorderWidth: 1,

        }},
        navigation: {{
        buttonOptions: {{
            enabled: false
        }}
    }},
        title: {{
                text: '{title}'
        }},
        credits:{{text:"",href:""}},
        xAxis: {{
                categories: {columns},
                labels:{{
                    style: {{
                    color: 'black',
                    fontSize:"large",
                    fontWeight:"bolder"
                    }}

                }},
                title: {{text:'{x_title}'}},
                opposite: true
                                
        }},
        yAxis: {{
                categories: {index},
                title: {{ text:'{y_title}'

                }}
        }},
        colorAxis: {{
                min: 0,
                minColor: '#FFFFFF',
                maxColor: '#FFFFFF'
        }},
        legend: {{
                enabled: false
        }},
        series: [{{
                name: '',
                borderWidth: 1,
                data: {data},
                dataLabels: {{
                        enabled: true,
                        color: '#000000'
                }}
        }}],
        tooltip: {{
        formatter: function () {{
                return this.point.value;
                }}
        }}
            }}"""

def serialize_table(df,title = '',output_type = 'notebook'):
    
#     df = df.loc[['All']+list(range(df.shape[0]-1,0,-1)),:]
    columns = list(df.columns)
    index = list(df.index)
    
    data_len = df.shape[0]
    width = 100 * df.shape[1]
    height = 30*data_len
    data = []
    for i,ii in enumerate(index):
        for c,cc in enumerate(columns):
            data.append([c,data_len-1-i,round(df.loc[ii,cc],4)])        
    index.reverse()
    json_chart = table_string_cross.format(
    columns = ["{0}".format(i) for i in columns],
    index =   ["{0}".format(i) for i in index],
    data = data,
    title = title,
    x_title = '' if not df.columns.name else df.columns.name,
    y_title   = '' if not df.index.name else df.index.name, 
        width=width,
        height = height
                         )
    if output_type == 'notebook':
        display_x_charts(json_chart)
    else:
        return json_chart
            
            
if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({"a":[1,2,3],"b":[4,6,3]})
    serialize(df, title="在notebook里测试",output_type='notebook') 
