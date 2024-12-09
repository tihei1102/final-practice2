import streamlit as st
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.tools import Tool
from langchain.schema import SystemMessage
from dotenv import load_dotenv
import constants as ct

load_dotenv()

st.markdown("## 問い合わせ対応自動化AIエージェント")
url = "https://fresh-cheque-38d.notion.site/AI-FAQ-89b471c62e5c4effb0f5500a9df92750"
st.markdown("※FAQページは[こちら](%s)" % url)

st.markdown(
    """<style>.stHorizontalBlock {
        flex-wrap: nowrap;
        margin-left: 56px;
        margin-top: -20px;
    }
    .stHorizontalBlock .stColumn:nth-of-type(2) {
        margin-left: -214px;
    }</style>""",
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

    st.session_state.feedback_yes_flg = False
    st.session_state.feedback_no_flg = False
    st.session_state.answer_flg = False
    st.session_state.dissatisfied_reason = ""
    st.session_state.feedback_no_reason_send_flg = False

    st.session_state.big_category_selected_flg = False
    st.session_state.small_category_selected_flg = False

    # 質問項目一覧と対応するフラグ立て
    st.session_state.contact_items = {}
    for big_category in ct.CONTACT_ITEMS:
        st.session_state.contact_items[big_category] = {
            "selected": False,  # 大カテゴリ単位でのフラグ
            "items": {}
        }
        for small_category in ct.CONTACT_ITEMS[big_category]["items"]:
            st.session_state.contact_items[big_category]["items"][small_category] = {
                "selected": False,  # 小カテゴリ単位でのフラグ
                "questions": {}
            }
            for index, _ in enumerate(ct.CONTACT_ITEMS[big_category]["items"][small_category]["questions"]):
                st.session_state.contact_items[big_category]["items"][small_category]["questions"][str(index)] = False # 質問単位でのフラグ

    # LLMとのやり取り
    system_template = """
    you are a helpful assistant.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=500,
        return_messages=True
    )
    st.session_state.chain = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )
    # general_tool = Tool.from_function(
    #     func=st.session_state.chain.run,
    #     name="一般的な質問への回答",
    #     description="一般的な質問に回答する"
    # )
    # tools = load_tools(["ddg-search"])
    # tools.append(general_tool)
    # st.session_state.agent_executor = initialize_agent(
    #     llm=llm,
    #     tools=tools,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    # )

with st.chat_message("assistant", avatar="images/robot.jpg"):
    st.success("私はお客様のご質問にお答えする生成AIロボットです。画面下部のチャット欄から対話形式でやり取りするか、以下のボタンをクリックしてよくある質問への回答をご参照ください。")
    for big_category in ct.CONTACT_ITEMS:
        if st.session_state.contact_items[big_category]["selected"]:
            st.session_state.contact_items[big_category]["selected"] = st.button(ct.CONTACT_ITEMS[big_category]["label"])
            st.session_state.contact_items[big_category]["selected"] = True
        else:
            st.session_state.contact_items[big_category]["selected"] = st.button(ct.CONTACT_ITEMS[big_category]["label"])
    # st.session_state.contact_flg = st.button("直接の問い合わせ")

input_message = st.chat_input("質問を入力してください。")

if st.session_state.feedback_no_flg and input_message:
    st.session_state.feedback_no_flg = False

for index, message in enumerate(st.session_state.messages):
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="images/ロボット.jpg"):
            st.markdown(message["content"])
            if index == len(st.session_state.messages) - 1:
                if st.session_state.feedback_yes_flg:
                    st.caption("ご満足いただけて良かったです！他にもご質問があれば、お気軽にお尋ねください！")
                    st.session_state.feedback_yes_flg = False
                    # 「はい」が押された場合の処理
                    # 1. 問い合わせログ一覧にお役立ち情報として登録するため、ChromaのDBに情報を保存する
                    # 2. 管理者向けのダッシュボード画面で、問い合わせ情報の一覧を表示する際、お役立ち情報として表示する
                if st.session_state.feedback_no_flg:
                    st.caption("ご期待に添えず申し訳ございません。今後の改善のために、差し支えない範囲でご満足いただけなかった理由を教えていただけますと幸いです。")
                    st.session_state.dissatisfied_reason = st.text_area("", label_visibility="collapsed")
                    if st.button("送信"):
                        st.session_state.feedback_no_flg = False
                        st.session_state.feedback_no_reason_send_flg = True
                        # 「いいえ」が押された場合の処理
                        # 1. 問い合わせログ一覧に不満足情報として登録するため、ChromaのDBに情報を保存する
                        # 2. 管理者向けのダッシュボード画面で、問い合わせ情報の一覧を表示する際、不満足情報として表示する
                        st.rerun()
                if st.session_state.feedback_no_reason_send_flg:
                    st.session_state.feedback_no_reason_send_flg = False
                    st.caption("ご回答いただき誠にありがとうございます。")
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if index == len(st.session_state.messages) - 1:
                if st.session_state.feedback_yes_flg:
                    st.caption("ご満足いただけて良かったです！他にもご質問があれば、お気軽にお尋ねください！")
                    st.session_state.feedback_yes_flg = False
                    # 「はい」が押された場合の処理
                    # 1. 問い合わせログ一覧にお役立ち情報として登録するため、ChromaのDBに情報を保存する
                    # 2. 管理者向けのダッシュボード画面で、問い合わせ情報の一覧を表示する際、お役立ち情報として表示する
                if st.session_state.feedback_no_flg:
                    st.caption("ご期待に添えず申し訳ございません。今後の改善のために、差し支えない範囲でご満足いただけなかった理由を教えていただけますと幸いです。")
                    st.session_state.dissatisfied_reason = st.text_area("", label_visibility="collapsed")
                    if st.button("送信"):
                        st.session_state.feedback_no_flg = False
                        st.session_state.feedback_no_reason_send_flg = True
                        # 「いいえ」が押された場合の処理
                        # 1. 問い合わせログ一覧に不満足情報として登録するため、ChromaのDBに情報を保存する
                        # 2. 管理者向けのダッシュボード画面で、問い合わせ情報の一覧を表示する際、不満足情報として表示する
                        st.rerun()
                if st.session_state.feedback_no_reason_send_flg:
                    st.session_state.feedback_no_reason_send_flg = False
                    st.caption("ご回答いただき誠にありがとうございます。")

for big_category in st.session_state.contact_items:
    if st.session_state.contact_items[big_category]["selected"]:
        with st.chat_message("assistant", avatar="images/ロボット.jpg"):
            st.markdown(f"「{ct.CONTACT_ITEMS[big_category]['label']}」の質問項目を選択してください。")
            for small_category in ct.CONTACT_ITEMS[big_category]["items"]:
                if st.session_state.contact_items[big_category]["items"][small_category]["selected"]:
                    st.session_state.contact_items[big_category]["items"][small_category]["selected"] = st.button(ct.CONTACT_ITEMS[big_category]["items"][small_category]["label"])
                    st.session_state.contact_items[big_category]["items"][small_category]["selected"] = True
                else:
                    st.session_state.contact_items[big_category]["items"][small_category]["selected"] = st.button(ct.CONTACT_ITEMS[big_category]["items"][small_category]["label"])
        click_flg = False
        for small_category in ct.CONTACT_ITEMS[big_category]["items"]:
            if st.session_state.contact_items[big_category]["items"][small_category]["selected"]:
                click_flg = True
                small_category_name = ct.CONTACT_ITEMS[big_category]["items"][small_category]["label"]
        if click_flg:
            with st.chat_message("assistant", avatar="images/ロボット.jpg"):
                st.markdown(f"「{small_category_name}」の質問項目を選択してください。")
                for small_category in ct.CONTACT_ITEMS[big_category]["items"]:
                    if st.session_state.contact_items[big_category]["items"][small_category]["selected"]:
                        for question_id, _ in enumerate(ct.CONTACT_ITEMS[big_category]["items"][small_category]["questions"]):
                            if st.session_state.contact_items[big_category]["items"][small_category]["questions"][str(question_id)]:
                                st.session_state.contact_items[big_category]["items"][small_category]["questions"][str(question_id)] = st.button(ct.CONTACT_ITEMS[big_category]["items"][small_category]["questions"][question_id]["question"])
                                st.session_state.contact_items[big_category]["items"][small_category]["questions"][str(question_id)] = True
                            else:
                                st.session_state.contact_items[big_category]["items"][small_category]["questions"][str(question_id)] = st.button(ct.CONTACT_ITEMS[big_category]["items"][small_category]["questions"][question_id]["question"])
            click_flg = False
            for small_category in ct.CONTACT_ITEMS[big_category]["items"]:
                for question_id, _ in enumerate(ct.CONTACT_ITEMS[big_category]["items"][small_category]["questions"]):
                    if st.session_state.contact_items[big_category]["items"][small_category]["questions"][str(question_id)]:
                        click_flg = True
                        answer = ct.CONTACT_ITEMS[big_category]["items"][small_category]["questions"][question_id]["answer"]
            if click_flg:
                with st.chat_message("assistant", avatar="images/ロボット.jpg"):
                    st.markdown(answer)

if input_message:
    st.session_state.messages.append({"role": "user", "content": input_message})
    with st.chat_message("user"):
        st.markdown(input_message)

    with st.spinner('回答生成中...'):
        result = st.session_state.chain.predict(input=input_message)
        # result = st.session_state.agent_executor.run(input_message)
        if input_message == "製品番号SHL-0459-0MC5Tのセキュリティ要件を教えて":
            result = """
                製品番号SHL-0459-0MC5Tのセキュリティ要件は次のように定義されています。\nこの製品は、高度なセキュリティ機能を備えており、特に以下の点で企業や個人のデータ保護を確保することを目的としています。\n\n**1. データ暗号化:**  \n本製品は、業界標準であるAES-256暗号化を採用しており、通信中および保存時のデータを暗号化します。この技術により、データの盗聴や改ざんを防ぎます。また、暗号化キーは動的に管理され、一定期間ごとに更新される仕組みが備わっています。\n\n**2. 認証プロセス:**  \n本製品は、二要素認証（2FA）をサポートしており、従来のパスワード認証に加えて、セキュリティトークンや生体認証（指紋または顔認識）を利用可能です。この仕組みにより、不正なログインを防ぐことができます。\n\n**3. アクセス制御:**  \nユーザーの役割に応じたロールベースアクセス制御（RBAC）を適用しています。これにより、管理者、一般ユーザー、ゲストなどの異なる権限レベルを設定し、不要なデータアクセスを防止します。さらに、本製品はISO 27001に準拠しており、第三者機関によるセキュリティ監査を定期的に受けています。詳細な技術仕様については、セキュリティ要件ドキュメント（ドキュメントID: SEC0459）をご参照ください。
            """
        elif input_message == "この製品は海外でも利用可能ですか？また海外への発送は可能ですか？":
            result = """
                この製品は海外でも問題なく利用可能です。ただし、製品の動作環境や電圧に依存する場合がありますので、利用される国の電圧やプラグ規格が製品に対応していることを事前にご確認ください。\n\nまた、海外への発送は現在50カ国以上に対応しており、主に北米、ヨーロッパ、アジア地域への配送を行っています。配送方法には標準配送とエクスプレス配送があり、選択可能です。具体的な発送手続きについては、当社ウェブサイトの『海外配送ガイド』をご参照いただくか、カスタマーサポートにお問い合わせください。
            """
        elif input_message == "支払い方法にはどんな選択肢がありますか？":
            result = """
                当社では、さまざまな支払い方法をご用意しております。以下が現在ご利用可能な支払い方法の一覧です。

                **1. クレジットカード:**  
                VISA、MasterCard、American Express、JCBなど主要なクレジットカードに対応しています。カード情報 を安全に保護するため、業界標準の暗号化技術を使用しています。

                **2. デビットカード:**  
                銀行口座と直接連動するデビットカードもご利用いただけます。クレジットカードと同じ手続きで簡単に支払い可能です。

                **3. 銀行振込:**  
                請求書発行後、指定の銀行口座にお振り込みいただく方法です。振込手数料はお客様のご負担となります。

                **4. 電子マネー・デジタルウォレット:** 
                PayPay、LINE Pay、楽天ペイ、Apple Pay、Google Payなどに対応しております。スマートフォンを使用した簡単な決済が可能です。

                **5. 後払いサービス:**  
                決済代行会社を通じた後払いも選択可能です。利用可能額や審査条件は各サービス提供会社によります。

                これらの方法は、オンラインストアおよび一部の実店舗で利用可能です。さらに詳細な手続きについては、当社の支払いガイドをご確認いただくか、カスタマーサポートまでお問い合わせください。
            """
        elif input_message == "FAQやマニュアルはどこで確認できますか？":
            result = """
                当社のFAQやマニュアルは、複数の方法でご確認いただけます。

                **1. ウェブサイト:**  
                公式ウェブサイトの『サポート』ページにアクセスすると、FAQやダウンロード可能なマニュアルが一覧で表示されます。検索バーを使えば、特定のキーワードやトピックで絞り込むことができます。
                
                **2. 製品パッケージ内のQRコード:**  
                お手元の製品パッケージに印刷されているQRコードをスキャンすると、該当製品のマニュアルやFAQページに直接アクセスできます。
                
                **3. サポートアプリ:**  
                当社のサポートアプリをスマートフォンにインストールすると、FAQや製品マニュアルを簡単に閲覧できるほか、最新のアップデート情報もご確認いただけます。
                
                **4. メールでのお問い合わせ:**  
                必要に応じて、カスタマーサポート（support@example.com）にご連絡いただければ、関連するリンクや資料を直接お送りすることも可能です。

                お客様が必要とする情報が確実に見つかるよう、さまざまなアクセス手段をご用意しております。
            """
        st.session_state.messages.append({"role": "assistant", "content": result})

    with st.chat_message("assistant", avatar="images/ロボット.jpg"):
        st.markdown(result)
        st.session_state.answer_flg = True
        st.caption("この回答はお役に立ちましたか？フィードバックをいただくことで、生成AIの回答の質が向上します。")

if st.session_state.answer_flg:
    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        if st.button("はい"):
            st.session_state.answer_flg = False
            st.session_state.feedback_yes_flg = True
            st.rerun()
    with col2:
        if st.button("いいえ"):
            st.session_state.answer_flg = False
            st.session_state.feedback_no_flg = True
            st.rerun()