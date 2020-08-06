from pymysql import connect,cursors

db_user = 'warn_algorithm'
db_pwd = 'Ps_warn#12'
db_host = '10.10.10.143'
db_port = 3306
db = 'bdmc'


mysql_conn = connect(user=db_user, host=db_host, database=db, password=db_pwd, port=db_port)
cursor = mysql_conn.cursor()