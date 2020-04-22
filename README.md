# Deltares-Climate-Change

<p align="center">
  <img width="300" height="300" src="heatmap.png">
</p>

Warning: Since the dataset Deltares provided us with is too big to just upload here, I recommend to store the data locally in a separate folder or ignore the specific directory (see https://stackoverflow.com/questions/343646/ignoring-directories-in-git-repositories-on-windows) to prevent uploading the data whenever you push your local changes.

Update: I created a .gitignore file. If you download the data and save it into a folder called Datares then you should now be able to keep it in the Deltares-Climate-Change folder and git will safely ignore it i.e. not try to upload it.


Okay so here is a quick Git tutorial in case you are using the command line:

    git status For checking (locally) what files have changed

    git add . For adding things (in this case everything because of the ".") to the staging area (locally)

    git commit For committing (i.e. writing into the (local!) database) everything that has been added to the staging area

    git push For uploading the committed stuff to the server
    
    git stash For a quick way of getting rid of all your local changes (but still being able to get them back later). Use this in case you want to have exactly the files which are online, but are not able to pull for some reason. Git stash will put all your local changes aside, which will help you pulling the available files. Note that this will get rid of your uncommitted changes, but in case you want them back there is a way to recover them (see https://www.git-scm.com/docs/git-stash)

