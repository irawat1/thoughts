---
layout: home
author_profile: true
---

# Welcome to My Data Science Portfolio

I'm a passionate data scientist and analyst who loves exploring data through Jupyter notebooks. This portfolio showcases my projects, analyses, and insights.

## Featured Notebooks

Here are some of my recent Jupyter notebook projects:

{% for notebook in site.notebooks limit:3 %}
- [{{ notebook.title }}]({{ notebook.url }}) - {{ notebook.excerpt | default: notebook.description }}
{% endfor %}

## About Me

I specialize in data analysis, machine learning, and creating interactive visualizations. My work focuses on extracting meaningful insights from complex datasets and presenting them in an accessible way.

## Get in Touch

Feel free to explore my [notebooks](/notebooks/), check out my [projects](/projects/), or [contact me](/contact/) if you'd like to collaborate!