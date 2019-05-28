from collections import defaultdict
import os
import tensorflow as tf
import logging
import os
from collections import defaultdict
import numpy as np


class Trainer(object):
    def __init__(self):
        pass

    @staticmethod
    def _train_sess(model, batch_generator, steps, summary_writer, save_summary_steps):
        global_step = tf.train.get_or_create_global_step()

        for i in range(steps):
            train_batch = batch_generator.next()
            train_batch["training"] = True
            feed_dict = {ph: train_batch[key] for key, ph in model.input_placeholder_dict.items() if key in train_batch}
            if i % save_summary_steps == 0:
                _, _, loss_val, summ, global_step_val = model.session.run([model.train_op, model.train_update_metrics,
                                                                           model.loss, model.summary_op, global_step],
                                                                          feed_dict=feed_dict)
                if summary_writer is not None:
                    summary_writer.add_summary(summ, global_step_val)
            else:
                _, _, loss_val = model.session.run([model.train_op, model.train_update_metrics, model.loss],
                                                   feed_dict=feed_dict)
            if np.isnan(loss_val):
                raise ValueError("NaN loss!")

        metrics_values = {k: v[0] for k, v in model.train_metrics.items()}
        metrics_val = model.session.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        logging.info("- Train metrics: " + metrics_string)

    @staticmethod
    def _eval_sess(model, batch_generator, steps, summary_writer):
        global_step = tf.train.get_or_create_global_step()

        final_output = defaultdict(list)
        for _ in range(steps):
            eval_batch = batch_generator.next()
            eval_batch["training"] = False
            feed_dict = {ph: eval_batch[key] for key, ph in model.input_placeholder_dict.items() if key in eval_batch}
            _, output = model.session.run([model.eval_update_metrics, model.output_variable_dict], feed_dict=feed_dict)
            for key in output.keys():
                final_output[key] += [v for v in output[key]]

        # Get the values of the metrics
        metrics_values = {k: v[0] for k, v in model.eval_metrics.items()}
        metrics_val = model.session.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        logging.info("- Eval metrics: " + metrics_string)

        # Add summaries manually to writer at global_step_val
        if summary_writer is not None:
            global_step_val = model.session.run(global_step)
            for tag, val in metrics_val.items():
                summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
                summary_writer.add_summary(summ, global_step_val)

        return final_output


    @staticmethod
    def _train_and_evaluate(model, train_batch_generator, eval_batch_generator, evaluator, epochs=1, eposides=1,
                            save_dir=None, summary_dir=None, save_summary_steps=10):
        best_saver = tf.train.Saver(max_to_keep=1) if save_dir is not None else None
        train_summary = tf.summary.FileWriter(os.path.join(summary_dir, 'train_summaries')) if summary_dir else None
        eval_summary = tf.summary.FileWriter(os.path.join(summary_dir, 'eval_summaries')) if summary_dir else None

        best_eval_score = 0.0
        for epoch in range(epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, epochs))
            train_batch_generator.init()
            train_num_steps = (
                                      train_batch_generator.get_instance_size() + train_batch_generator.get_batch_size() - 1) // train_batch_generator.get_batch_size()
            model.session.run(model.train_metric_init_op)

            # one epoch consists of several eposides
            assert isinstance(eposides, int)
            num_steps_per_eposide = (train_num_steps + eposides - 1) // eposides
            for eposide in range(eposides):
                logging.info("Eposide {}/{}".format(eposide + 1, eposides))
                current_step_num = min(num_steps_per_eposide, train_num_steps - eposide * num_steps_per_eposide)
                eposide_id = epoch * eposides + eposide + 1
                Trainer._train_sess(model, train_batch_generator, current_step_num, train_summary, save_summary_steps)

                if model.ema_decay>0:
                    trainable_variables = tf.trainable_variables()
                    cur_weights = model.session.run(trainable_variables)
                    model.session.run(model.restore_ema_variables)
                # Save weights
                if save_dir is not None:
                    last_save_path = os.path.join(save_dir, 'last_weights', 'after-eposide')
                    model.save(last_save_path, global_step=eposide_id)

                # Evaluate for one epoch on dev set
                eval_batch_generator.init()
                eval_instances = eval_batch_generator.get_instances()
                model.session.run(model.eval_metric_init_op)

                eval_num_steps = (eval_batch_generator.get_instance_size() + eval_batch_generator.get_batch_size() - 1) // eval_batch_generator.get_batch_size()
                output = Trainer._eval_sess(model, eval_batch_generator, eval_num_steps, eval_summary)
                # pred_answer = model.get_best_answer(output, eval_instances)
                score = evaluator.get_score(model.get_best_answer(output, eval_instances))

                metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in score.items())
                logging.info("- Eval metrics: " + metrics_string)
                
                if model.ema_decay>0:
                    feed_dict = {}
                    for i in range(len(trainable_variables)):
                        feed_dict[model.ema_placeholders[i]] = cur_weights[i]
                    model.session.run(model.restore_cur_variables, feed_dict=feed_dict)

                # Save best weights
                eval_score = score[evaluator.get_monitor()]
                if eval_score > best_eval_score:
                    logging.info("- epoch %d eposide %d: Found new best score: %f" % (epoch + 1, eposide + 1, eval_score))
                    best_eval_score = eval_score
                    # Save best weights
                    if save_dir is not None:
                        best_save_path = os.path.join(save_dir, 'best_weights', 'after-eposide')
                        best_save_path = best_saver.save(model.session, best_save_path, global_step=eposide_id)
                        logging.info("- Found new best model, saving in {}".format(best_save_path))

    @staticmethod
    def inference(model, batch_generator, steps):
        global_step = tf.train.get_or_create_global_step()
        final_output = defaultdict(list)
        for _ in range(steps):
            eval_batch = batch_generator.next()
            eval_batch["training"] = False
            feed_dict = {ph: eval_batch[key] for key, ph in model.input_placeholder_dict.items() if key in eval_batch and key not in ['answer_start','answer_end','is_impossible']}
            output = model.session.run(model.output_variable_dict, feed_dict=feed_dict)
            for key in output.keys():
                final_output[key] += [v for v in output[key]]
        return final_output


    @staticmethod
    def _evaluate(model, batch_generator, evaluator):
        # Evaluate for one epoch on dev set
        batch_generator.init()
        eval_instances = batch_generator.get_instances()

        eval_num_steps = (len(
            eval_instances) + batch_generator.get_batch_size() - 1) // batch_generator.get_batch_size()
        output = Trainer._eval_sess(model, batch_generator, eval_num_steps, None)
        pred_answer = model.get_best_answer(output, eval_instances)
        score = evaluator.get_score(pred_answer)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in score.items())
        logging.info("- Eval metrics: " + metrics_string)

    @staticmethod
    def _inference(model, batch_generator):
        batch_generator.init()
        model.session.run(model.eval_metric_init_op)
        instances = batch_generator.get_instances()
        eval_num_steps = (len(instances) + batch_generator.get_batch_size() - 1) // batch_generator.get_batch_size()
        output = Trainer.inference(model, batch_generator, eval_num_steps)
        pred_answers = model.get_best_answer(output, instances)
        return pred_answers

