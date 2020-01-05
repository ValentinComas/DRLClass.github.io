# DRL_class

Cours de Deep reinforcement learning, Université Lyon 1, M2 IA

<a href="ressources/M2IA_course_2019_2020.pdf">Slides du CM</a>

<a href="ressources/TP_DRL_2019_2020.pdf">Projet DRL</a>

## Commandes

Pour activer l'environnement virtuel :

```
pipenv shell #Créer et/ou active l’environnement virtuel du projet (a faire dans le dossier)
```

Pour installer les libs :

```
pip install -r .\requirements.txt
```

Pour lancer un entraînement :

```
python train.py model
```
où model est le nom du modèle qui va être sauvegardé (il sera précédé de "model" par défaut)
Par exemple, pour créer model_0 :
```
python train.py _0
```

Pour lancer les tests sur le modèle :

```
python test.py model nb_render
```
où model est le nom du modèle qui va être testé (il sera précédé de "model" par défaut) et nb_render le nombre d'épisodes dont on veut le rendu à la fin (par défaut 1)
Par exemple, pour tester model_0 sans rendu à la fin :
```
python train.py _0 0
```
Site d'aide : [ici](http://sametmax.com/pipenv-solution-moderne-pour-remplacer-pip-et-virtualenv/)
