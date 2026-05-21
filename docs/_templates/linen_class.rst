.. In the sphinx v3.0.4 source code:
   sphinx/ext/autosummary/templates/autosummary/class.rst

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:

    {% block methods %}

    {% if methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
    {% for item in all_methods %}
        {% if not item.startswith('_') or item in ['__init__',
                                                    '__len__',
                                                    '__call__',
                                                    '__iter__',
                                                    '__getitem__',
                                                    '__setitem__',
                                                    ] %}
        ~{{ name }}.{{ item }}
        {%- endif -%}
    {%- endfor %}
    {% endif %}

    {% endblock %}

    {% block attributes %}

    {% if attributes %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
    {% for item in attributes %}
        {% if not item.startswith('_') %}
        ~{{ name }}.{{ item }}
        {%- endif -%}

    {%- endfor %}
    {% endif %}

    {% endblock %}
