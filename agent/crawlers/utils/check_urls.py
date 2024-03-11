FORBIDDEN_URLS = ["facebook.com", "youtube.com", "instagram.com", "snopes.com", "politifact.com",
                  "cdc.gov", "who.int"]

FORBIDDEN_SUBSTRINGS = ["download", ".pdf", ".jpeg", ".jpg", ".png"]

def is_url_allowed(url):
    for forbidden_url in FORBIDDEN_URLS:
        if forbidden_url in url:
            return False
    return True

def is_url_ok(url):
    for forbidden_substring in FORBIDDEN_SUBSTRINGS:
        if forbidden_substring in url:
            return False
    return True


if __name__ == "__main__":
    url = "https://www.snopes.com/2020/03/20/snopes-on-covid-19-fact-checking/"

    print(is_url_allowed(url))