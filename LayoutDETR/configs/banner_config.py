"""
Copyright (c) 2023 Salesforce, Inc.

All rights reserved.

SPDX-License-Identifier: Apache License 2.0

For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/

By Chia-Chih Chen, chiachih.chen@salesforce.com
"""
class BannerConfig:
    MAX_IMG_WIDTH = 2000
    MAX_IMG_HEIGHT = 2000
    MAX_TIMG_WIDTH = 600  # thumbnail image
    MAX_TIMG_HEIGHT = 400
    MIN_BOX_BOX_MARGIN = 16
    MIN_BOX_IMG_MARGIN = 8
    MIN_BOX_JITTER_RATIO = 0.975
    MAX_BOX_JITTER_RATIO = 1.25

    LABEL_LIST = [
        'header',
        'pre-header',
        'post-header',
        'body',
        'disclaimer / footnote',
        'button',
        'callout',
        'logo'
    ]


class RendererConfig:
    BROWSER_CONFIG = ['no-sandbox', 'disable-infobars', 'disable-dev-shm-usage', 'disable-browser-side-navigation',
                      'disable-gpu', 'disable-features=VizDisplayCompositor', 'headless']
    TEXT_CSS_TEMP = "align-items:center;position:absolute;word-wrap:break-word;overflow-wrap:break-word;display:flex;"
    HTML_TEMP = \
        """
        <!DOCTYPE html>
        <html>
        <head>
        <link rel="stylesheet" type="text/css" href="css/style.css">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
        .container {
          position: relative;
          color: white;
        }
        .body {
          margin: 0;
          padding: 0;
        }
        </style>
        </head>
        <body class="body">
        <div class="container">
          <img src="" alt="" style="width:auto;position:absolute;top:0px;left:0px;">
        </div>
        </body>
        </html> 
        """

    TEXT_CSS_TEMP = 'align-items:center;position:absolute;word-wrap:break-word;overflow-wrap:' \
                    'break-word;display:flex;'
    BORDER_CSS_TEMP = ''
    TEXT_CSS_TEMP += BORDER_CSS_TEMP
    JAVASCRIPT = \
        """    
        let inner = document.getElementById("{}_in");
        let outer = document.getElementById("{}").style;
        if (inner && outer) {{
            inner.style.fontSize = "{}px"
            resize_to_fit()
        }}
        return inner.style.fontSize;

        function resize_to_fit() {{
            let fontSize = inner.style.fontSize;
            inner.style.fontSize = (parseFloat(fontSize) - 1) + 'px';
            if (inner.clientHeight >= parseFloat(outer.height) - {}) {{
                resize_to_fit();
            }}
        }}
        """