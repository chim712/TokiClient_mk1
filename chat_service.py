import requests

import tts


def chat():
    url = "http://127.0.0.1:8001"

    print("Start Chat Module")

    response = requests.post(url+"/session/start")
    sid = response.json()["session_id"]
    print(sid)

    headers = {
        "Content-Type": "application/json"
    }
    while True:
        tmpStr = input("메시지 입력: ")

        params = {"session_id":sid, "message":tmpStr}
        response = requests.post(url+"/chat/", params=params, headers=headers)
        # ---------
        print(response.status_code)
        print(response.text)

        if response.headers.get("Content-Type") == "application/json":
            res_json = response.json()
            tmpStr = res_json.get("response", "")
        else:
            print("서버 응답이 JSON이 아닙니다:", response.text)
            continue
            # ---------

        tmpStr = response.json()["response"]
        print(tmpStr)
        tts.speak_korean(tmpStr)

        if response.json()["response"] == "CloseChat":
            print("대화를 종료합니다.")
            break

    response = requests.post(url+"/session/complete", params={"session_id":sid}, headers=headers)

