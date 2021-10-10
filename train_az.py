import sys, os
data_dir = sys.argv[1]

print("===== DATA =====")
print("DATA PATH: " + data_dir)
print("LIST FILES IN DATA DIR...")
print(os.listdir(data_dir))
print("================")