## Building the Environment in the VM

* SSH to the EC2 instance using PuTTY (for windows) or directly using Terminal (for MAC)
* Install the ```awscli``` using ```sudo apt install awscli```.

### Installing Anaconda on the EC2

  * ```wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh```
  * ```bash Anaconda3-2020.02-Linux-x86_64.sh ```
  * ```source .bashrc```. After this the terminal should be in the anaconda environment. Type in, ```which python``` to view the path to the python folder in Anaconda.

* Next set the transfer accelaration endpoints as true and download the dataset using the below commands

  * ```aws configure set default.s3.use_accelerate_endpoint true```
  * ```aws s3 cp s3://mininuscenes/data.zip ./data.zip --endpoint-url https://mininuscenes.s3-accelerate.amazonaws.com```
 
* To download the entire nuscenes dataset use the following command

  * ```aws s3 cp s3://nuscenes-full/sweeps.tar.gz ./data/sweeps.tar.gz --endpoint-url https://nuscenes-full.s3-accelerate.amazonaws.com```
  * ```aws s3 cp s3://nuscenes-full/samples.tar.gz ./data/samples.tar.gz --endpoint-url https://nuscenes-full.s3-accelerate.amazonaws.com```
  * ```aws s3 cp s3://nuscenes-full/v1.0-test_meta.tgz ./data/v1.0-test_meta.tgz --endpoint-url https://nuscenes-full.s3-accelerate.amazonaws.com```

### Setting up the Jupyter Notebook to run on the EC2 VM.

  * To create a jupyter configuration file, Type in ```jupyter notebook --generate-config```.
  * A password for the notebook needs to be created which can be done using the IPython library.
  
  * Type in, ```ipython``` and type the below commands in the python prompt.
  
    * ```from IPython.lib import passwd```
    * ```passwd()```. Note: Save the has password as it is required while editing the configuration file
    * ```exit```
  
  * Add the following code to the configuration file created above.
    * Change directory to jupyter using ```cd .jupyter``` 
    * Edit the file using nano or vim using ```nano jupyter_notebook_config.py```
    * At the end of the file add the below code 
      * ```conf = get_config()```
      * ```conf.NotebookApp.ip = '0.0.0.0'```
      * ```conf.NotebookApp.password = u'HASH_PASSWORD'```
      * ```conf.NotebookApp.port = 8888```
      
  * Connecting the jupyter notebook to the EC2 server
  
    * Type in, ```jupyter notebook```
    * ```https://(your AWS public dns):8888/```
