"""Streamlit entry point of the LexGraph interface.

Start it with ``lexgraph ui`` (or ``streamlit run src/lexgraph/ui/app.py``). The interface
talks to the HTTP API at ``LEXGRAPH_API_URL``, sending ``LEXGRAPH_API_KEY`` when it is set.
"""

from functools import partial

import streamlit as st

from lexgraph.config import get_settings
from lexgraph.ui import about, chat, library
from lexgraph.ui.client import ApiClient


@st.cache_resource(show_spinner=False)
def default_client() -> ApiClient:
    """One API client per server process, configured from the ``LEXGRAPH_*`` settings."""
    settings = get_settings()
    api_key = settings.api_key.get_secret_value() if settings.api_key is not None else None
    return ApiClient(settings.api_url, api_key)


def main(client: ApiClient | None = None) -> None:
    """Configure the app and run the selected page; tests inject ``client``.

    Each page sets its own layout, so switching pages never inherits the previous one.
    """
    st.set_page_config(
        page_title="LexGraph",
        page_icon=":material/balance:",
        menu_items={"About": "LexGraph: perguntas e respostas sobre a legislação federal."},
    )
    api = client if client is not None else default_client()
    pages = [
        st.Page(
            partial(chat.render, api),
            title="Chat",
            icon=":material/forum:",
            url_path="chat",
            default=True,
        ),
        st.Page(
            partial(library.render, api),
            title="Acervo",
            icon=":material/menu_book:",
            url_path="acervo",
        ),
        st.Page(
            partial(about.render, api), title="Sobre", icon=":material/info:", url_path="sobre"
        ),
    ]
    st.navigation(pages, position="top").run()


if __name__ == "__main__":
    main()
