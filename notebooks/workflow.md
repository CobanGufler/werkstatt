WORKFLOW

git checkout main
git pull
git checkout -b feature/func1
   ... arbeiten ...
git add -A
git commit -m "feat: func1 added"
git push -u origin feature/func1

   Auf GitHub unter "Pull Request" den Request öffnen, bei Merge auf das Pfeil rechts klicken
   und "Squash & Merge" auswählen und branch deleten

git checkout main
git pull
git branch -d feature/func1        # lokalen Branch löschen


   Nächste Bearbeitung startet analog:
git checkout -b feature/nächstes-thema