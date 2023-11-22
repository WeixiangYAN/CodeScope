import os
import os.path as osp
import argparse
from subprocess import Popen, PIPE
import json
import func_timeout
from func_timeout import func_set_timeout
import ast

@func_set_timeout(5)
def record_result(output_dict, src_uid, submission_id, difficulty, id, answer, output, outerr, errtype=None):
    output_dict[submission_id] = {}
    output_dict[submission_id]["src_uid"] = src_uid
    output_dict[submission_id]["submission_id"] = submission_id
    if difficulty:
        output_dict[submission_id]["difficulty"] = difficulty
    if id:
        output_dict[submission_id]["id"] = id
    if answer:
        output_dict[submission_id]["answer"] = answer
    if output:
        output_dict[submission_id]["output"] = output
    if outerr:
        output_dict[submission_id]["error"] = outerr
    if errtype:
        output_dict[submission_id]["errtype"] = errtype
    return output_dict


@func_set_timeout(30)
def exe_testcase(source_code, answer, input, lang, postfix, output_dict, collapse_num, total_case, wrong_case, src_uid,
                 submission_id, difficulty, id, ):
    tmps_path = f"{args.project_path}\\{args.tmps_dir}\\"

    if not osp.exists(tmps_path):
        os.mkdir(tmps_path)

    record, err = 0, 0
    outlog, outerr, errtype = None, None, None

    if lang == "d":
        p = Popen(f'cd "{args.d_path}"', shell=True)
        try:
            cmmond_line = f'"{args.cmd_path}" /c rdmd.exe "{args.project_path}\\temp.d"'

            p = Popen(cmmond_line, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
            p.stdin.write(input.encode())
            p.stdin.flush()

            output, outerr = p.communicate()
            output = output.decode()
            if outerr:
                outerr = outerr.decode()
                errtype = "RUNTIME_ERROR"
            total_case += 1

            answer = answer.replace("\r", "")
            answer = answer.replace("\r\n", "\n")
            output = output.replace("\r", "")
            output = output.replace("\r\n", "\n")

            output = output.replace(" ", "").lower().strip()
            answer = answer.replace(" ", "").lower().strip()
        except Exception as e:
            print(e, "runtime error in src_uid: ", src_uid)
            err = 1
            errtype = "COMPILATION_ERROR"
            output_dict["error"] = record_result(output_dict["error"], src_uid, submission_id, difficulty, id, answer,
                                                 None, outerr, errtype)
            record = 1
            wrong_case += 1

    elif lang == "delphi":
        file_path = tmps_path + str(collapse_num) + '_temp.' + postfix
        collapse_flag = 0

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(source_code)

        try:
            p = Popen(f'cd "{args.delphi_path}"', shell=True)
            cmmond_line = f'"{args.cmd_path}" /c DCC32.EXE "{tmps_path}{collapse_num}_temp.dpr"'
            p = Popen(cmmond_line, stdout=PIPE, shell=True)
            p.wait(1)
            outlog = p.stdout.read()
            if outlog:
                outlog = outlog.decode(encoding="utf-8")

        except Exception as e:
            print(e, "compilation error in src_uid: ", src_uid)
            collapse_num += 1
            collapse_flag = 1
            errtype = "COMPILATION_ERROR"

        try:
            cmmond_line = f'{args.cmd_path} /c "{tmps_path}' + str(collapse_num) + '_temp.exe"'
            p = Popen(cmmond_line, stdin=PIPE, stdout=PIPE, shell=True)
            p.stdin.write(input.encode())
            p.stdin.flush()
            p.wait(1)

            os.remove(tmps_path + str(collapse_num) + "_temp.exe")

        except Exception as e:
            print(e, "RUNTIME_ERROR in src_uid: ", src_uid)
            collapse_num += 1
            collapse_flag = 1
            if errtype is None:
                errtype = "RUNTIME_ERROR"

        try:
            output, outerr = p.communicate()
            if output is not None:
                output = output.decode(encoding="utf-8")
                answer = answer.replace("\r", "")
                answer = answer.replace("\r\n", "\n")
                output = output.replace("\r", "")
                output = output.replace("\r\n", "\n")

                output = output.replace(" ", "").lower().strip()
                answer = answer.replace(" ", "").lower().strip()
            if outerr is not None:
                outerr = outerr.decode(encoding="utf-8")
            total_case += 1

        except Exception as e:
            print(e, 'record error', src_uid)
            if not collapse_flag:
                collapse_num += 1
            wrong_case += 1
            err = 1
            if outerr:
                outerr = outerr.decode(encoding="utf-8")
                if outlog:
                    outerr += outlog
            else:
                if outlog:
                    outerr = outlog

            if errtype is None:
                errtype = "RUNTIME_ERROR"
            output_dict["error"] = record_result(output_dict["error"], src_uid, submission_id, difficulty, id, None,
                                                 None, outerr, errtype)
            record = 1

    elif lang == "perl":
        try:
            p = Popen('perl temp.pl', stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
            p.stdin.write(input.encode())
            p.stdin.flush()

            output, outerr = p.communicate()
            output = output.decode(encoding="utf-8")
            if outerr:
                outerr = outerr.decode(encoding="utf-8")

            total_case += 1

            answer = answer.replace("\r", "")
            answer = answer.replace("\r\n", "\n")
            output = output.replace("\r", "")
            output = output.replace("\r\n", "\n")

            output = output.replace(" ", "").lower().strip()
            answer = answer.replace(" ", "").lower().strip()
        except Exception as e:
            print(e, "Runtime_error src_uid: ", src_uid)
            err = 1
            errtype = "RUNTIME_ERROR"
            output_dict["error"] = record_result(output_dict["error"], src_uid, submission_id, difficulty, id, None,
                                                 None, outerr, errtype)
            wrong_case += 1
            collapse_num += 1
    if err == 0:
        if answer != output and output != "":
            print("-----------------answer: ", answer, "-------------------")
            print("-----------------output: ", output, "-------------------")
            print("WRONG_ANSWER in src_uid: ", src_uid)
            errtype = "WRONG_ANSWER"
            try:
                output_dict["wrong"] = record_result(output_dict["wrong"], src_uid, submission_id, difficulty, id,
                                                     answer, output, outerr, errtype)
            except func_timeout.exceptions.FunctionTimedOut:
                print("Time Limit Exceeded")
                if outlog and lang == "delphi":
                    if outerr is not None:
                        outerr += outlog
                output_dict["error"] = record_result(output_dict["error"], src_uid, submission_id, difficulty, id, None,
                                                     None, outerr, errtype)
            wrong_case += 1
            err = 1
        elif answer != output and output == "":
            print("RUNTIME_ERROR")
            errtype = "RUNTIME_ERROR"
            if outlog and lang == "delphi":
                if outerr is not None:
                    outerr += outlog
                else:
                    outerr = outlog
            output_dict["error"] = record_result(output_dict["error"], src_uid, submission_id, difficulty, id, None,
                                                 None, outerr, errtype)
            wrong_case += 1
            err = 1
    else:
        if record == 0:
            if outlog and lang == "delphi":
                outerr += outlog
            else:
                outerr = outlog
            errtype = "RUNTIME_ERROR"
            output_dict["error"] = record_result(output_dict["error"], src_uid, submission_id, difficulty, id, None,
                                                 None, outerr, errtype)

    return output_dict, wrong_case, err, collapse_num


@func_set_timeout(300)
def exe_question(content, lang, postfix, output_dict, collapse_num):
    source_code = content["source_code"]

    id = None
    if "id" in content.keys():
        id = content["id"]
    src_uid = str(content["src_uid"])
    difficulty = str(content["difficulty"])
    testcases = ast.literal_eval(content['testcases'])
    if "code_uid" in content.keys():
        submission_id = str(content["code_uid"])
    else:
        submission_id = str(content["submission_id"])

    if source_code == "":
        print("No source code detected")
        output_dict["error"] = record_result(output_dict["error"], src_uid, submission_id, difficulty, id, None, None,
                                             "No Source Code", "No_Source_Code")
        return output_dict, 1, collapse_num

    source_code = source_code.replace("\\\\", "\\")
    source_code = source_code.replace("\\r", "\r")
    source_code = source_code.replace("\\n", "\n")
    source_code = source_code.replace("\\\"", "\"")
    source_code = source_code.replace("\r", "")
    source_code = source_code.replace("\r\n", "\n")

    file_path = 'temp.' + postfix
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(source_code)


    total_case, wrong_case = 0, 0
    for testcase in testcases:
        input = testcase["input"][0]
        answer = testcase["output"][0]

        input = input.replace("\r", "")
        input = input.replace("\r\n", "\n")

        try:
            output_dict, wrong_case, err, collapse_num = exe_testcase(source_code, answer, input, lang, postfix,
                                                                      output_dict, collapse_num, total_case, wrong_case,
                                                                      src_uid, submission_id, difficulty, id)
        except func_timeout.exceptions.FunctionTimedOut:
            err, wrong_case = 1, 1
            print("Time Limit Exceeded")
            collapse_num += 1
            output_dict["error"] = record_result(output_dict["error"], src_uid, submission_id, difficulty, id, None,
                                                 None, "Time Limit Exceeded", "RUNTIME_ERROR")

        if err == 1:
            wrong_case = 1
            break
    if err == 0:
        output_dict["accepted"] = record_result(output_dict["accepted"], src_uid, submission_id, difficulty, id, None,
                                                None, None, None)

    return output_dict, wrong_case, collapse_num


def exe_main():

    jsonl_path = args.jsonl_path
    lang = jsonl_path.split(".")[0].split("_")[-1]
    if lang == "d":
        postfix = "d"
    elif lang == "delphi":
        postfix = "dpr"
    elif lang == "perl":
        postfix = "pl"
    else:
        print("Don't support this language")

    code_sum, correct_sum, collapse_num = 0, 0, 0
    output_dict = {"accepted": {}, "wrong": {}, "error": {}}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            content = json.loads(line)
            try:
                output_dict, wrong_case, collapse_num = exe_question(content, lang, postfix,
                                                                     output_dict, collapse_num)
            except func_timeout.exceptions.FunctionTimedOut:
                print("Time Limit Exceeded")
                wrong_case = 1

            code_sum += 1
            if wrong_case == 0:
                correct_sum += 1

            print("done: ", idx + 1, " not accepted: ", idx + 1 - correct_sum)

    wrong_num = len(output_dict["wrong"].keys())
    error_num = len(output_dict["error"].keys())
    print("code_sum:", code_sum, " correct_sum: ", correct_sum, " wrong_num: ", wrong_num, " error_num: ", error_num,
          " accurancy: ", correct_sum / code_sum)
    output_dict["info"] = {"code_sum": code_sum, "correct_sum": correct_sum, "wrong_num": wrong_num, "error_num":
        error_num, "accurancy": correct_sum / code_sum}

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmps_dir', type=str, default="tmps")
    parser.add_argument('--code_path', type=str, default="program_synthesis_eval_palm_d.jsonl")
    parser.add_argument('--delphi_path', type=str, help="Your delphi bin directory", required=False)
    parser.add_argument('--d_path', type=str, help="Your d(dmd) bin directory", required=False)
    parser.add_argument('--cmd_path', type=str, help="Your cmd.exe")
    parser.add_argument('--project_path', type=str, default="..\\..\\..",  help="Root directory")
    parser.add_argument('--output_path', type=str, default=".\\results\\executed_result.json")

    args = parser.parse_args()

    exe_main()

