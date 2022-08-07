# face_id
this app is a face recognition app for face id usage.
use `http://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf` book for reference  <br />
<br />
<br />
<br />


use `appendix.html` file in the data directory to download 2 dependencies: <br />
    - `siamesemodel.h5` --> the model trained and save in this file <br />
    - `lfw.tgz` --> the negative data in this file and after download this unzip in the data directory <br />
    <br />
    <br />
    <br />


to run MODEL to train it by your own data you can run the `main_model.py` python script. <br />
    please run at `face_id` directory: <br />
    run: `python src/MODEL/main_model.py` <br />
    <br />
    <br />

to run face_id app you can run `face_id_app.py` python script. <br />
    please run at `face_id` directory: <br />
    run: `python src/APP/face_id_app.py` <br />
