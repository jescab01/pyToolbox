# -*- coding: utf-8 -*-
#
# https://bitybyte.github.io/Organzando-codigo-Python/
#
# ¿Qué pongo en el archivo __init__.py?
# En general no es necesario poner nada en el archivo __init__.py, pero es muy común usarlo para realizar configuraciones e importar cualquier objeto necesario de nuestra librería.
#
# Por ejemplo, en nuestro ejemplo, si el archivo archivo1.py contiene una clase llamada Archivo, podemos importarla con __init__.py para que esté disponible al nivel de paquete. Normalmente para importar esta clase, tendríamos que hacer lo siguiente:
#
# from package.archivo1 import Archivo
# Pero podemos simplificarlo con nuestro __init__.py así:
#
# # En el archivo package/__init__.py
# from archivo1 import Archivo
#
# # En tu programa que utiliza el paquete package
# from package import Archivo
# Otra posibilidad bastanté útil de __init__.py es la variable __all__. Esta variable contiene la lista de módulos que serán importados al utilizar import *. Para nuestra estructura de ejemplo, si el __init__.py del directorio subpackage/ contiene lo siguiente:
#
# __all__ = ['elmodulo1','elmodulo2']
# Entonces al realizar lo siguiente:
#
# from subpackage import *
# # Importa elmodulo1 y elmodulo2
# Con estas herramientas puedes hacer que tus paquetes sean mucho más elegantes de importar y manejar. Ojalá haya resultado útil, y cualquier duda o comentario, adelante!