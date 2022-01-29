"""
    File Name: mpld3_example.py
    Date: 6/6/2020
    Updated:
    Author: reed.clarke@rcsoftwareservices.com
"""

import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins


class HelloWorld(plugins.PluginBase):  # inherit from PluginBase
    """Hello World plugin"""

    JAVASCRIPT = """
    mpld3.register_plugin("helloworld", HelloWorld);
    HelloWorld.prototype = Object.create(mpld3.Plugin.prototype);
    HelloWorld.prototype.constructor = HelloWorld;
    function HelloWorld(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    HelloWorld.prototype.draw = function(){
        this.fig.canvas.append("text")
            .text("hello world")
            .style("font-size", 72)
            .style("opacity", 0.3)
            .style("text-anchor", "middle")
            .attr("x", this.fig.width / 2)
            .attr("y", this.fig.height / 2)
    }
    """

    def __init__(self):
        self.dict_ = {"type": "helloworld"}


fig, ax = plt.subplots()
plugins.connect(fig, HelloWorld())
mpld3.show()
# print(mpld3.fig_to_html(fig, template_type="simple"))
