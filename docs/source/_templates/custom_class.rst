{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :private-members: _default_saver, _default_loader

   .. ^^ is for documenting these private methods of the Model class

   {% block methods %}

   {% set _methods = methods | reject("in", inherited_members) | reject("eq", "__init__") | list %}
   {% if _methods | length > 0  %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:

   {% for item in _methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   .. special cases for certain classes
   {% set _attributes = attributes %}
   {% if module == "unionml.schedule" and objname == "Schedule" %}
      {% set _attributes = ["type", "name", "expression", "offset", "fixed_rate", "time_arg", "inputs", "activate_on_deploy", "launchplan_kwargs"] %}
   {% endif %}


   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
