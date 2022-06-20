import os
import argparse
import time
from datetime import date
from do_gp import do_gp
import pandas as pd
import matplotlib.pyplot as plt
import s3fs
import os.path

# replaced by command line argument
TASK_ID = int(os.environ["SGE_TASK_ID"]) - 1
#JOBID = int(os.environ["JOB_ID"])


date = str(time.strftime("%d-%m-%Y"))

config = [
    dict(species="dog",
     mpc="gastroenteric",
     location = None,
     plot_filename="dog_gi",
     plot_title="Dog Records labelled as Gastroenteric main presenting complaint Nationwide \n as of " + date,
    ),
    dict(species="dog",
     mpc="pruritus",
     location = None,
     plot_filename="dog_pruritus",
     plot_title="Dog Records labelled as Pruritus main presenting complaint Nationwide \n as of " + date,
    ),
    dict(species="dog",
     mpc="respiratory",
     location = None,
     plot_filename="dog_respiratory",
     plot_title="Dog Records labelled as Respiratory main presenting complaint Nationwide \n as of " + date,
    ),
    dict(species="dog",
     mpc="gastroenteric",
     location="Yorkshire and The Humber",
     plot_filename="dog_gi_yorkshire",
     plot_title="Dog Records labelled as Gastroenteric main presenting complaint in Yorkshire \n as of " + date,
    ),
    dict(species="dog",
     mpc="gastroenteric",
     location="South East (England)",
     plot_filename="dog_gi_southeast",
     plot_title="Dog Records labelled as Gastroenteric main presenting complaint in South East \n as of " + date,
    ),
    dict(species="dog",
     mpc="gastroenteric",
     location="London",
     plot_filename="dog_gi_London",
     plot_title="Dog Records labelled as Gastroenteric main presenting complaint in London \n as of " + date,
    ),
    dict(species="dog",
     mpc="gastroenteric",
     location="South West (England)",
     plot_filename="dog_gi_southwest",
     plot_title="Dog Records labelled as Gastroenteric main presenting complaint in South West \n as of " + date,
    ),
    dict(species="dog",
     mpc="gastroenteric",
     location="East",
     plot_filename="dog_gi_east",
     plot_title="Dog Records labelled as Gastroenteric main presenting complaint in East \n as of " + date,
    ),
    dict(species="dog",
     mpc="gastroenteric",
     location="East Midlands (England)",
     plot_filename="dog_gi_eastmidlands",
     plot_title="Dog Records labelled as Gastroenteric main presenting complaint in East Midlands \n as of " + date,
    ),
    dict(species="dog",
     mpc="gastroenteric",
     location="Wales",
     plot_filename="dog_gi_wales",
     plot_title="Dog Records labelled as Gastroenteric main presenting complaint in Wales \n as of " + date,
    ),
    dict(species="dog",
     mpc="gastroenteric",
     location="Scotland",
     plot_filename="dog_gi_scotland",
     plot_title="Dog Records labelled as Gastroenteric main presenting complaint in Scotland \n as of " + date,
    ),
    dict(species="dog",
     mpc="gastroenteric",
     location="Northern Ireland",
     plot_filename="dog_gi_northernireland",
     plot_title="Dog Records labelled as Gastroenteric main presenting complaint in Northern Ireland \n as of " + date,
    ),
    dict(species="dog",
     mpc="gastroenteric",
     location="North West (England)",
     plot_filename="dog_gi_northwest",
     plot_title="Dog Records labelled as Gastroenteric main presenting complaint in North West \n as of " + date,
    ),
    dict(species="dog",
     mpc="gastroenteric",
     location="North East (England)",
     plot_filename="dog_gi_northeast",
     plot_title="Dog Records labelled as Gastroenteric main presenting complaint in North East \n as of " + date,
    ),
    dict(species="dog",
     mpc="gastroenteric",
     location="West Midlands (England)",
     plot_filename="dog_gi_westmidlands",
     plot_title="Dog Records labelled as Gastroenteric main presenting complaint in West Midlands \n as of " + date,
    ),
    dict(species="dog",
     mpc="respiratory",
     location="Yorkshire and The Humber",
     plot_filename="dog_respiratory_yorkshire",
     plot_title="Dog Records labelled as Respiratory main presenting complaint in Yorkshire \n as of " + date,
    ),
    dict(species="dog",
     mpc="respiratory",
     location="South East (England)",
     plot_filename="dog_respiratory_southeast",
     plot_title="Dog Records labelled as Respiratory main presenting complaint in South East \n as of " + date,
    ),
    dict(species="dog",
     mpc="respiratory",
     location="London",
     plot_filename="dog_respiratory_London",
     plot_title="Dog Records labelled as Respiratory main presenting complaint in London \n as of " + date,
    ),
    dict(species="dog",
     mpc="respiratory",
     location="South West (England)",
     plot_filename="dog_gi_southwest",
     plot_title="Dog Records labelled as Respiratory main presenting complaint in South West \n as of " + date,
    ),
    dict(species="dog",
     mpc="respiratory",
     location="East",
     plot_filename="dog_respiratory_east",
     plot_title="Dog Records labelled as Respiratory main presenting complaint in East \n as of " + date,
    ),
    dict(species="dog",
     mpc="respiratory",
     location="East Midlands (England)",
     plot_filename="dog_respiratory_eastmidlands",
     plot_title="Dog Records labelled as Respiratory main presenting complaint in East Midlands \n as of " + date,
    ),
    dict(species="dog",
     mpc="respiratory",
     location="Wales",
     plot_filename="dog_respiratory_wales",
     plot_title="Dog Records labelled as Respiratory main presenting complaint in Wales \n as of " + date,
    ),
    dict(species="dog",
     mpc="respiratory",
     location="Scotland",
     plot_filename="dog_respiratory_scotland",
     plot_title="Dog Records labelled as Respiratory main presenting complaint in Scotland \n as of " + date,
    ),
    dict(species="dog",
     mpc="respiratory",
     location="Northern Ireland",
     plot_filename="dog_respiratory_northernireland",
     plot_title="Dog Records labelled as Respiratory main presenting complaint in Northern Ireland \n as of " + date,
    ),
    dict(species="dog",
     mpc="respiratory",
     location="North West (England)",
     plot_filename="dog_respiratory_northwest",
     plot_title="Dog Records labelled as Respiratory main presenting complaint in North West \n as of " + date,
    ),
    dict(species="dog",
     mpc="respiratory",
     location="North East (England)",
     plot_filename="dog_respiratory_northeast",
     plot_title="Dog Records labelled as Respiratory main presenting complaint in North East \n as of " + date,
    ),
    dict(species="dog",
     mpc="respiratory",
     location="West Midlands (England)",
     plot_filename="dog_respiratory_westmidlands",
     plot_title="Dog Records labelled as Respiratory main presenting complaint in West Midlands \n as of " + date,
    ),
    dict(species="dog",
     mpc="pruritus",
     location="Yorkshire and The Humber",
     plot_filename="dog_pruritus_yorkshire",
     plot_title="Dog Records labelled as Pruritus main presenting complaint in Yorkshire \n as of " + date,
    ),
    dict(species="dog",
     mpc="pruritus",
     location="South East (England)",
     plot_filename="dog_pruritus_southeast",
     plot_title="Dog Records labelled as Pruritus main presenting complaint in South East \n as of " + date,
    ),
    dict(species="dog",
     mpc="pruritus",
     location="London",
     plot_filename="dog_pruritus_London",
     plot_title="Dog Records labelled as Pruritus main presenting complaint in London \n as of " + date,
    ),
    dict(species="dog",
     mpc="pruritus",
     location="South West (England)",
     plot_filename="dog_pruritus_southwest",
     plot_title="Dog Records labelled as Pruritus main presenting complaint in South West \n as of " + date,
    ),
    dict(species="dog",
     mpc="pruritus",
     location="East",
     plot_filename="dog_pruritus_east",
     plot_title="Dog Records labelled as Pruritus main presenting complaint in East \n as of " + date,
    ),
    dict(species="dog",
     mpc="pruritus",
     location="East Midlands (England)",
     plot_filename="dog_pruritus_eastmidlands",
     plot_title="Dog Records labelled as Pruritus main presenting complaint in East Midlands \n as of " + date,
    ),
    dict(species="dog",
     mpc="pruritus",
     location="Wales",
     plot_filename="dog_pruritus_wales",
     plot_title="Dog Records labelled as Pruritus main presenting complaint in Wales \n as of " + date,
    ),
    dict(species="dog",
     mpc="pruritus",
     location="Scotland",
     plot_filename="dog_pruritus_scotland",
     plot_title="Dog Records labelled as Pruritus main presenting complaint in Scotland \n as of " + date,
    ),
    dict(species="dog",
     mpc="pruritus",
     location="Northern Ireland",
     plot_filename="dog_pruritus_northernireland",
     plot_title="Dog Records labelled as Pruritus main presenting complaint in Northern Ireland \n as of " + date,
    ),
    dict(species="dog",
     mpc="pruritus",
     location="North West (England)",
     plot_filename="dog_pruritus_northwest",
     plot_title="Dog Records labelled as pruritus main presenting complaint in North West \n as of " + date,
    ),
    dict(species="dog",
     mpc="pruritus",
     location="North East (England)",
     plot_filename="dog_pruritus_northeast",
     plot_title="Dog Records labelled as Pruritus main presenting complaint in North East \n as of " + date,
    ),
    dict(species="dog",
     mpc="pruritus",
     location="West Midlands (England)",
     plot_filename="dog_pruritus_westmidlands",
     plot_title="Dog Records labelled as Pruritus main presenting complaint in West Midlands \n as of " + date,
    ),
    dict(species="cat",
     mpc="gastroenteric",
     location = None,
     plot_filename="cat_gi",
     plot_title="Cat Records labelled as Gastroenteric main presenting complaint Nationwide \n as of " + date,
    ),
    dict(species="cat",
     mpc="pruritus",
     location = None,
     plot_filename="cat_pruritus",
     plot_title="Cat Records labelled as Pruritus main presenting complaint Nationwide \n as of " + date,
    ),
    dict(species="cat",
     mpc="respiratory",
     location = None,
     plot_filename="cat_respiratory",
     plot_title="Cat Records labelled as Respiratory main presenting complaint Nationwide \n as of " + date,
    ),
    dict(species="cat",
     mpc="gastroenteric",
     location="Yorkshire and The Humber",
     plot_filename="cat_gi_yorkshire",
     plot_title="Cat Records labelled as Gastroenteric main presenting complaint in Yorkshire \n as of " + date,
    ),
    dict(species="cat",
     mpc="gastroenteric",
     location="South East (England)",
     plot_filename="cat_gi_southeast",
     plot_title="Cat Records labelled as Gastroenteric main presenting complaint in South East \n as of " + date,
    ),
    dict(species="cat",
     mpc="gastroenteric",
     location="London",
     plot_filename="cat_gi_London",
     plot_title="Cat Records labelled as Gastroenteric main presenting complaint in London \n as of " + date,
    ),
    dict(species="cat",
     mpc="gastroenteric",
     location="South West (England)",
     plot_filename="cat_gi_southwest",
     plot_title="Cat Records labelled as Gastroenteric main presenting complaint in South West \n as of " + date,
    ),
    dict(species="cat",
     mpc="gastroenteric",
     location="East",
     plot_filename="cat_gi_east",
     plot_title="Cat Records labelled as Gastroenteric main presenting complaint in East \n as of " + date,
    ),
    dict(species="cat",
     mpc="gastroenteric",
     location="East Midlands (England)",
     plot_filename="cat_gi_eastmidlands",
     plot_title="Cat Records labelled as Gastroenteric main presenting complaint in East Midlands \n as of " + date,
    ),
    dict(species="cat",
     mpc="gastroenteric",
     location="Wales",
     plot_filename="cat_gi_wales",
     plot_title="cat Records labelled as Gastroenteric main presenting complaint in Wales \n as of " + date,
    ),
    dict(species="cat",
     mpc="gastroenteric",
     location="Scotland",
     plot_filename="cat_gi_scotland",
     plot_title="Cat Records labelled as Gastroenteric main presenting complaint in Scotland \n as of " + date,
    ),
    dict(species="cat",
     mpc="gastroenteric",
     location="Northern Ireland",
     plot_filename="cat_gi_northernireland",
     plot_title="Cat Records labelled as Gastroenteric main presenting complaint in Northern Ireland \n as of " + date,
    ),
    dict(species="cat",
     mpc="gastroenteric",
     location="North West (England)",
     plot_filename="cat_gi_northwest",
     plot_title="Cat Records labelled as Gastroenteric main presenting complaint in North West \n as of " + date,
    ),
    dict(species="cat",
     mpc="gastroenteric",
     location="North East (England)",
     plot_filename="cat_gi_northeast",
     plot_title="Cat Records labelled as Gastroenteric main presenting complaint in North East \n as of " + date,
    ),
    dict(species="cat",
     mpc="gastroenteric",
     location="West Midlands (England)",
     plot_filename="cat_gi_westmidlands",
     plot_title="Cat Records labelled as Gastroenteric main presenting complaint in West Midlands \n as of " + date,
    ),
    dict(species="cat",
     mpc="respiratory",
     location="Yorkshire and The Humber",
     plot_filename="cat_respiratory_yorkshire",
     plot_title="Cat Records labelled as Respiratory main presenting complaint in Yorkshire \n as of " + date,
    ),
    dict(species="cat",
     mpc="respiratory",
     location="South East (England)",
     plot_filename="cat_respiratory_southeast",
     plot_title="Cat Records labelled as Respiratory main presenting complaint in South East \n as of " + date,
    ),
    dict(species="cat",
     mpc="respiratory",
     location="London",
     plot_filename="cat_respiratory_London",
     plot_title="Cat Records labelled as Respiratory main presenting complaint in London \n as of " + date,
    ),
    dict(species="cat",
     mpc="respiratory",
     location="South West (England)",
     plot_filename="cat_gi_southwest",
     plot_title="Cat Records labelled as Respiratory main presenting complaint in South West \n as of " + date,
    ),
    dict(species="cat",
     mpc="respiratory",
     location="East",
     plot_filename="cat_respiratory_east",
     plot_title="Cat Records labelled as Respiratory main presenting complaint in East \n as of " + date,
    ),
    dict(species="cat",
     mpc="respiratory",
     location="East Midlands (England)",
     plot_filename="cat_respiratory_eastmidlands",
     plot_title="Cat Records labelled as Respiratory main presenting complaint in East Midlands \n as of " + date,
    ),
    dict(species="cat",
     mpc="respiratory",
     location="Wales",
     plot_filename="cat_respiratory_wales",
     plot_title="Cat Records labelled as Respiratory main presenting complaint in Wales \n as of " + date,
    ),
    dict(species="cat",
     mpc="respiratory",
     location="Scotland",
     plot_filename="cat_respiratory_scotland",
     plot_title="Cat Records labelled as Respiratory main presenting complaint in Scotland \n as of " + date,
    ),
    dict(species="cat",
     mpc="respiratory",
     location="Northern Ireland",
     plot_filename="cat_respiratory_northernireland",
     plot_title="Cat Records labelled as Respiratory main presenting complaint in Northern Ireland \n as of " + date,
    ),
    dict(species="cat",
     mpc="respiratory",
     location="North West (England)",
     plot_filename="cat_respiratory_northwest",
     plot_title="Cat Records labelled as Respiratory main presenting complaint in North West \n as of " + date,
    ),
    dict(species="cat",
     mpc="respiratory",
     location="North East (England)",
     plot_filename="cat_respiratory_northeast",
     plot_title="Cat Records labelled as Respiratory main presenting complaint in North East \n as of " + date,
    ),
    dict(species="cat",
     mpc="respiratory",
     location="West Midlands (England)",
     plot_filename="cat_respiratory_westmidlands",
     plot_title="Cat Records labelled as Respiratory main presenting complaint in West Midlands \n as of " + date,
    ),
    dict(species="cat",
     mpc="pruritus",
     location="Yorkshire and The Humber",
     plot_filename="cat_pruritus_yorkshire",
     plot_title="Cat Records labelled as Pruritus main presenting complaint in Yorkshire \n as of " + date,
    ),
    dict(species="cat",
     mpc="pruritus",
     location="South East (England)",
     plot_filename="cat_pruritus_southeast",
     plot_title="Cat Records labelled as Pruritus main presenting complaint in South East \n as of " + date,
    ),
    dict(species="cat",
     mpc="pruritus",
     location="London",
     plot_filename="cat_pruritus_London",
     plot_title="Cat Records labelled as Pruritus main presenting complaint in London \n as of " + date,
    ),
    dict(species="cat",
     mpc="pruritus",
     location="South West (England)",
     plot_filename="cat_pruritus_southwest",
     plot_title="Cat Records labelled as Pruritus main presenting complaint in South West \n as of " + date,
    ),
    dict(species="cat",
     mpc="pruritus",
     location="East",
     plot_filename="cat_pruritus_east",
     plot_title="Cat Records labelled as Pruritus main presenting complaint in East \n as of " + date,
    ),
    dict(species="cat",
     mpc="pruritus",
     location="East Midlands (England)",
     plot_filename="cat_pruritus_eastmidlands",
     plot_title="Cat Records labelled as Pruritus main presenting complaint in East Midlands \n as of " + date,
    ),
    dict(species="cat",
     mpc="pruritus",
     location="Wales",
     plot_filename="cat_pruritus_wales",
     plot_title="Cat Records labelled as Pruritus main presenting complaint in Wales \n as of " + date,
    ),
    dict(species="cat",
     mpc="pruritus",
     location="Scotland",
     plot_filename="cat_pruritus_scotland",
     plot_title="Cat Records labelled as Pruritus main presenting complaint in Scotland \n as of " + date,
    ),
    dict(species="cat",
     mpc="pruritus",
     location="Northern Ireland",
     plot_filename="cat_pruritus_northernireland",
     plot_title="Cat Records labelled as Pruritus main presenting complaint in Northern Ireland \n as of " + date,
    ),
    dict(species="cat",
     mpc="pruritus",
     location="North West (England)",
     plot_filename="cat_pruritus_northwest",
     plot_title="Cat Records labelled as pruritus main presenting complaint in North West \n as of " + date,
    ),
    dict(species="cat",
     mpc="pruritus",
     location="North East (England)",
     plot_filename="cat_pruritus_northeast",
     plot_title="Cat Records labelled as Pruritus main presenting complaint in North East \n as of " + date,
    ),
    dict(species="cat",
     mpc="pruritus",
     location="West Midlands (England)",
     plot_filename="cat_pruritus_westmidlands",
     plot_title="Cat Records labelled as Pruritus main presenting complaint in West Midlands \n as of " + date,
    )
    ]


if __name__ == "__main__":
    """Drives production of GP timeseries analyses.

    USAGE: python gp_array.py -d <raw dataset> -r <region dataset> \
                              -l <lockdown dataset> -i <task id>

    From CLI: $ python gp_array.py -h
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d",
                        type=str,
                        required=False,
                        help="The path of the raw dataset")
    parser.add_argument("--region", "-r", type=str, required=False)
    parser.add_argument("--lockdown", "-l", type=str, required=False)
    parser.add_argument("--taskid", "-i", type=int, default=1)

    args = parser.parse_args()


    
if args.dataset == None:
    s3 = s3fs.S3FileSystem(profile='agile-upload', client_kwargs={'endpoint_url':'https://fhm-chicas-storage.lancs.ac.uk'})
    files = s3.glob('savsnet-agile-upload/mpc*', detail=True)
    sorted_files = sorted(files.values(),key=lambda x: x['LastModified'], reverse=True)
    print("Opening dir:", sorted_files[0]['Key'])
    with s3.open(sorted_files[0]['Key'], 'rb') as f:
        rawdataset = pd.read_json(f, compression='gzip')
else:
    rawdataset = pd.read_json(args.dataset, compression='gzip')

if args.region == None:
    s3 = s3fs.S3FileSystem(profile='agile-upload', client_kwargs={'endpoint_url':'https://fhm-chicas-storage.lancs.ac.uk'})
    files = 'localauthoritycodes.csv'
    bucket = 'savsnet-agile-upload'
    regiondataset = pd.read_csv(s3.open('{}/{}'.format(bucket, files), mode='rb'))
else:
    regiondataset = pd.read_csv(args.region)


if args.lockdown == None:
    s3 = s3fs.S3FileSystem(profile='agile-upload', client_kwargs={'endpoint_url':'https://fhm-chicas-storage.lancs.ac.uk'})
    files = 'lockdowncountryedited.csv'
    bucket = 'savsnet-agile-upload'
    lockdowndataset = pd.read_csv(s3.open('{}/{}'.format(bucket, files), mode='rb'))
else:
    lockdowndataset = pd.read_csv(args.lockdown)


mindate = pd.to_datetime('2019-01-01')
fig,ax=plt.subplots(figsize=(10,5))



do_gp(rawdataset,
          lockdowndataset,
          regiondataset,
          config[TASK_ID],
          mindate,
          ax)

os.environ['SGE_STDOUT_PATH']


