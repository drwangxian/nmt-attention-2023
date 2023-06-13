import tensorflow as tf

aa = '¿Todavía está en casa?'
tb_summary_writer = tf.summary.create_file_writer('tb-utf8')
with tb_summary_writer.as_default():
    tf.summary.text('utf-8', aa, step=0)
tb_summary_writer.close()