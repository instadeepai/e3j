.. In the sphinx v3.0.4 source code:
   sphinx/ext/autosummary/templates/autosummary/class.rst

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{# per-class members hidden from both the autoclass body and the summary table #}
{% set hidden = {
    'TensorProduct': ['aggregation_method', 'target_matrix', 'aggregate'],
    'Bigotimes': ['aggregation_method', 'target_matrix', 'aggregate'],
}.get(objname, []) %}

.. autoclass:: {{ objname }}
    :members:
    :inherited-members:
    {%- if hidden %}
    :exclude-members: {{ hidden | join(', ') }}
    {%- endif %}

    {% block methods %}

    {% if methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
    {% for item in all_methods %}
        {% if (not item.startswith('_') or item in ['__init__',
                                                    '__len__',
                                                    '__call__',
                                                    '__iter__',
                                                    '__getitem__',
                                                    '__setitem__',
                                                    ]) and item not in hidden %}
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
        {% if not item.startswith('_') and item not in hidden %}
        ~{{ name }}.{{ item }}
        {%- endif -%}

    {%- endfor %}
    {% endif %}

    {% endblock %}
