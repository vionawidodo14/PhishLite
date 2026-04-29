import urllib
from IPy import IP
import re
from tld import get_tld, get_fld, is_tld
import statistics
from urllib.parse import urlparse
import requests
from datetime import datetime
import subprocess
import json
import logging
import numpy as np
import joblib
import pickle
import whois
from urllib.parse import urlparse, urljoin

import time
from collections import defaultdict

FEATURE_TIMINGS = defaultdict(list)

def time_feature(name, func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    FEATURE_TIMINGS[name].append(elapsed)
    return result

# URL based features
def checkLength(URL):
    """
    Phishers can use long URL to hide the doubtful part in the address bar.
    if len(url)>75->phishing,<54->legit，else:0 suspicious
    """
    legitimate_threshold = 54
    suspicious_threshold = 75
    # return(len(URL))
    if len(URL) < legitimate_threshold:
        return -1
    elif len(URL) < suspicious_threshold:
        return 0
    else:
        return 1


def hexDecoder(domain):
    """
    Function that inspects a given domain to check if it is a hex-encoded IP address.
    If the domain is an hex-encoded IP address, it return the IPv4 address;
    if the domain is not an IP address, it returns 0
    """
    try:
        n = domain.split(".")
        IPv4 = str(int(n[0], 16))
        for number in n[1:]:
            IPv4 = IPv4 + "." + str(int(number, 16))
        return IPv4
    except:
        return 0


def checkIP(URL):
    """
    Function that inspects a given URL to determine if it contains an IP address.
    If returns 1 if it is an IP address, and -1 otherwise
    """
    if URL.count(".") == 1 and URL.startswith("http") is False:
        domain = URL
    else:
        domain = (urlparse(URL)).netloc

    try:
        i = IP(domain)
        # print("{} is a valid IP address".format(domain))
        i = 1
    except:
        decoded = hexDecoder(domain)
        if decoded == 0:
            i = -1
            # print("{} is not a valid IP address".format(domain))
            return i
        try:
            i = IP(decoded)
            # print("{} is an IP address in hexadecimal format".format(domain))
            i = 1
        except Exception as e:
            print(e)
            i = -1
    return i


def checkRedirect(URL):
    """
    The existence of “//” within the URL path means that the user will be redirected to another website.
    An example of such URL’s is: “http://www.legitimate.com//http://www.phishing.com”.phishing:1,legit:-1
    """
    if URL.rfind("//") > 7:
        redirect = 1
    else:
        redirect = -1
    return redirect


def checkShortener(URL):
    """
    check if the url is a tiny url.
    """
    shorteners_list = [
        "bit.do",
        "t.co",
        "lnkd.in",
        "db.tt",
        "qr.ae",
        "adf.ly",
        "goo.gl",
        "bitly.com",
        "cur.lv",
        "tinyurl.com",
        "ow.ly",
        "bit.ly",
        "ity.im",
        "q.gs",
        "is.gd",
        "po.st",
        "bc.vc",
        "twitthis.com",
        "u.to",
        "j.mp",
        "buzurl.com",
        "cutt.us",
        "u.bb",
        "yourls.org",
        "x.co",
        "prettylinkpro.com",
        "scrnch.me",
        "filoops.info",
        "vzturl.com",
        "qr.net",
        "1url.com",
        "tweez.me",
        "v.gd",
        "tr.im",
        "link.zip.net",
        "tinyarrows.com",
        "adcraft.co",
        "adcrun.ch",
        "adflav.com",
        "aka.gr",
        "bee4.biz",
        "cektkp.com",
        "dft.ba",
        "fun.ly",
        "fzy.co",
        "gog.li",
        "golinks.co",
        "hit.my",
        "id.tl",
        "linkto.im",
        "lnk.co",
        "nov.io",
        "p6l.org",
        "picz.us",
        "shortquik.com",
        "su.pr",
        "sk.gy",
        "tota2.com",
        "xlinkz.info",
        "xtu.me",
        "yu2.it",
        "zpag.es",
    ]
    for s in shorteners_list:
        if URL.find(s + "/") > -1:
            return 1
    return -1


def checkSubdomains(URL):
    """
    If the number of dots (aside from the "WWW" and the "ccTLD") is greater than one,
    then the URL is classified as “Suspicious” since it has one sub domain.
    However, if the dots are greater than two, it is classified as “Phishing” since it will have multiple sub domains.
    else if the number of dots is one,no sub domain->legit
    """
    if URL.count(".") == 1 and URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            domain = URL[:ind]
        else:
            domain = URL
    else:
        domain = ((urlparse(URL)).netloc).lower()

    if domain.startswith("www."):
        domain = domain[4:]
    counter = domain.count(".") - 2
    if counter > 0:
        return 1  # phish
    elif counter == 0:
        return 0  # suspicious
    else:
        return -1  # legit


def checkAt(URL):
    """
    Using “@” symbol in the URL leads the browser to ignore everything preceding the “@” symbol
    and the real address often follows the “@” symbol.
    """
    if URL.find("@") >= 0:
        at = 1
    else:
        at = -1
    return at


def checkFakeHTTPS(URL):
    """
    The phishers may add the “HTTPS” token to the domain part of a URL in order to trick users. For example,
    http://https-www-paypal-it-webapps-mpp-home.soft-hair.com/.
    """

    if URL.count(".") == 1 and URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            domain = URL[:ind]
        else:
            domain = URL

    else:
        domain = ((urlparse(URL)).netloc).lower()

    if domain.find("https") > -1:
        return 1
    else:
        return -1


def checkDash(URL):
    """
    The dash symbol is rarely used in legitimate URLs.
    For example http://www.Confirme-paypal.com/.
    """
    if URL.count(".") == 1 and URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            domain = URL[:ind]
        else:
            domain = URL
    else:
        domain = ((urlparse(URL)).netloc).lower()

    if domain.find("-") > -1:
        return 1
    else:
        return -1


def checkDataURI(URL):
    """
    Function that determines if the URL is a DataURI.
    It returns 1 if it's a DataURI, and -1 otherwise.
    """
    if URL.startswith("data:"):
        return 1
    return -1


def checkNumberofCommonTerms(URL):
    """
    check the freqency of common terms "http,//,.com,www",usually they appear only once in legit webpage.
    """
    url = URL.lower()
    common_term = ["http", "www", ".com", "//"]
    for term in common_term:
        if url.count(term) > 1:
            return 1
        else:
            continue
    return -1


def checkNumerical(URL):
    """
    Numerical characters are uncommon for benign domains and especially subdomains in our dataset.
    """
    try:
        res = get_tld(URL, as_object=True)
    except:
        return 1
    domain = res.subdomain + res.domain
    number = re.search(r"\d+", domain)
    if number:
        return 1
    else:
        return -1


def checkPathExtend(URL):
    """
    Malicious scripts can be added to legitimate pages. Some file extensions used in
    URL paths may lunch such kind of attacks. Presence of the following malicious
    path extensions is considered: ’txt’, ’exe’, ’js’

    """
    extension = [".txt", ".exe", ".js"]
    if URL.count(".") == 1 and URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            path = URL[ind:]
        else:
            path = None

    else:
        path = (urlparse(URL).path).lower()

    if path:
        for ex in extension:
            if path.find(ex) > -1:
                return 1

    return -1


def checkPunycode(URL):
    """
    Punycode is used in domain names to replace some ASCIIs with Unicode，check if the domain include punycode.
    """
    if URL.count(".") == 1 and URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            domain = URL[:ind]
        else:
            domain = URL
    else:
        domain = ((urlparse(URL)).netloc).lower()

    subdomain = domain.split(".")

    for i in subdomain:
        mat = re.search("^xn--[a-z0-9]{1,59}|-$", i)
        if mat:
            return 1

    return -1


def checkSensitiveWord(URL):
    """
    check how many sensitive words in the url
    """
    sensitive_words = [
        "secure",
        "account",
        "webscr",
        "login",
        "ebayisapi",
        "signin",
        "banking",
        "confirm",
    ]
    counts = 0
    for word in sensitive_words:
        num = URL.count(word)

        counts = counts + num
    return counts


def checkTLDinPath(URL):
    """
    In well-formed URLs, top-level domains (TLDs) appear only before the path.
    """
    try:
        res = get_tld(URL, as_object=True, fix_protocol=True)
    except:
        # print("[INFO] Top-level domain not found")
        return 1
    path = res.parsed_url.path
    if path:
        path = path.lower().split(".")
        for pa in path:
            if is_tld(pa):
                return 1
    return -1


def checkTLDinSub(URL):
    """
    In well-formed URLs, top-level domains (TLDs) appear only before the path.
    When TLDs in the subdomain part, the URL is considered as phishing.
    """
    try:
        res = get_tld(URL, as_object=True, fix_protocol=True)
    except:
        # print("[INFO] Top-level domain not found")
        return 1
    sub_domain = res.subdomain
    if sub_domain:
        sub = sub_domain.lower().split(".")
        for s in sub:
            if is_tld(s):
                return 1
    return -1


def totalWordUrl(URL):
    """
    NLP feature, the number of words
    """
    res = re.split(r"[/:\.?=\&\-\s\_]+", URL)
    total = len(res)
    return total


"""
Natural language processing and word-raw features are also used in phishing
detection. We consider number of words, char repeat, shortest
words in URLs, hostnames, and paths, longest words in URLs, hostnames, and paths, average length of words in URLs,
hostnames, and paths.
"""


def shortestWordUrl(URL):
    """
    the length of shortest word in url
    """
    res = re.split(r"[/:\.?=\&\-\s\_]+", URL)

    try:
        shortest = min((word for word in res if word), key=len)
        return len(shortest)
    except:
        return 0


def shortestWordHost(URL):
    """
    the length of shortest word in hostname
    """
    hostname = urlparse(URL).netloc
    res = hostname.split(".")
    try:
        shortest = min((word for word in res if word), key=len)
        return len(shortest)
    except:
        return 0


def shortestWordPath(URL):
    """
    the length of shortest word in path
    """
    if URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            path = URL[ind:]
        else:
            path = None
    else:
        path = (urlparse(URL).path).lower()

    res = re.split(r"[/:\.?=\&\-\s\_]+", path)
    try:
        shortest = min((word for word in res if word), key=len)
        return len(shortest)
    except:
        return 0


def longestWordUrl(URL):
    """
    the length of longest word in URL
    """
    res = re.split(r"[/:\.?=\&\-\s\_]+", URL)
    try:
        longest = max((word for word in res if word), key=len)
    except:
        return 0
    return len(longest)


def longestWordHost(URL):
    """
    the length of longest word in host
    """
    if URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            hostname = URL[:ind]
        else:
            hostname = URL
    else:
        hostname = urlparse(URL).hostname

    res = re.split(r"[/:\.?=\&\-\s\_]+", hostname)

    try:
        longest = max((word for word in res if word), key=len)
        return len(longest)
    except:
        return 0


def longestWordPath(URL):
    """
    the length of longest word in path
    """
    if URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            path = URL[ind:]
        else:
            path = None
    else:
        path = (urlparse(URL).path).lower()

    res = re.split(r"[/:\.?=\&\-\s\_]+", path)

    try:
        longest = max((word for word in res if word), key=len)
        return len(longest)
    except:
        return 0


def averageWordUrl(URL):
    """
    average length of words in url
    """
    res = re.split(r"[/:\.?=\&\-\s\_]+", URL)
    average = statistics.mean((len(word) for word in res if word))
    return float(format(average, ".2f"))


def averageWordHost(URL):
    """
    average length of words in host
    """
    if URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            hostname = URL[:ind]
        else:
            hostname = URL
    else:
        hostname = urlparse(URL).hostname

    res = re.split(r"[/:\.?=\&\-\s\_]+", hostname)
    average = statistics.mean((len(word) for word in res if word))
    return float(format(average, ".2f"))


def averageWordPath(URL):
    """
    average length of words in path
    """
    if URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            path = URL[ind:]
        else:
            path = None
    else:
        path = (urlparse(URL).path).lower()

    res = re.split(r"[/:\.?=\&\-\s\_]+", path)

    try:
        average = statistics.mean((len(word) for word in res if word))
        return float(format(average, ".2f"))
    except:
        return 0


def checkStatisticRe(URL):
    """
    collect top10 phishing domain from phishtank, the url is more likely to be a phishing site if its firstdoamin
    or top domain in this list

    """
    top_fdomains = [
        "esy.es",
        "hol.es",
        "000webhostapp.com",
        "for-our.info",
        "bit.ly",
        "16mb.com",
        "96.lt",
        "totalsolution.com.br",
        "beget.tech",
        "sellercancelordernotification.com",
    ]
    top_tld = [
        "surf",
        "cn",
        "bid",
        "gq",
        "ml",
        "cf",
        "work",
        "cam",
        "ga",
        "casa",
        "tk",
        "ga",
        "top",
        "cyou",
        "bar",
        "rest",
    ]
    try:
        f_domain = get_fld(URL)
        t_domain = get_tld(URL)
    except:
        return 1

    for f in top_fdomains:
        if f_domain.find(f) > -1:
            return 1
        else:
            for t in top_tld:
                if t_domain.find(t) > -1:
                    return 0
    return -1


# get url features from third-party services





def checkPR(URL):
    """git
    page rank(1-10) indicate whether the webpage is popular, legit page usually has a high rank.
    we use Openpagerank to get the webpage's page rank.
    """
    try:
        domain = get_fld(URL)
    except:
        return 0
    # print('domain',domain)
    headers = {
        "API-OPR": "c48080g840k0wc8cw88g0o40w4gg4kcksgs00k8k"
    }  # api of Openpagerank,https://www.domcop.com/openpagerank/what-is-openpagerank
    url = "https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D=" + domain
    request = requests.get(url, headers=headers)
    result = request.json()
    try:
        resp = result["response"]
    except:
        return 0

    for item in resp:
        pr = item["page_rank_integer"]
    return pr


def getWhois(URL):
    """
    get url's whois info
    """
    try:
        who = whois.whois(URL)
    except:
        who = None
    return who


def checkDNS(who):
    """
    legit webpage has whois information.
    """
    if who is None:
        return 1
    try:
        domain_name = who["domain_name"]
    except:
        try:
            domain_name = who["domain"]
        except:
            return 1
    if (domain_name) is None:
        return 1
    return -1


def checkRegistrationLen(who):
    """
    Function that checks if the domain age of the website is suspicious or not.
    It considers the length of the period between the domain creation and its expiration dates (from the WHOIS query).
    It returns 1 if the length cannot be computed, or if the length is shorter than "age_threshold"; and -1 otherwise.

    """
    age_threshold = 364
    if who is None:
        return 1
    try:
        creation = who["creation_date"][0]
        expiration = who["expiration_date"][0]
    except:
        return 1

    length = (expiration - creation).days
    if length > age_threshold:
        return -1
    return 1


def checkAge(URL, who):
    """
    Function that checks if the domain age of the website is suspicious or not.
    It considers the length of the period between the domain creation and today (from the WHOIS query).
    It returns 1 if the length cannot be computed, or if the length is shorter than "age_threshold"; and -1 otherwise.
    """
    if who is None:
        return 1
    age_threshold = 180
    try:
        creation = who["creation_date"][0]  # 1999-10-11 11:05:17
        now = datetime.now()
    except:
        return 1
    try:
        age = (now - creation).days
        if age > age_threshold:
            return -1
    except:
        return 1
    return 1


def checkAbnormal(who, URL):
    """
    Function that compares the identity of a website with the record in the whois database.

    """
    if URL.count(".") == 1 and URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            domain = URL[:ind]
        else:
            domain = URL
    else:
        domain = ((urlparse(URL)).netloc).lower()
    if whois is None:
        return 1
    try:
        domain_name = who["domain_name"]
    except:
        try:
            domain_name = who["domain"]
        except:
            return 1
    if (domain_name) is None:
        return 1
    if len(domain_name[0]) == 1:
        if domain == (domain_name.lower()):
            return -1
    else:
        for d in domain_name:
            if domain == d.lower():
                return -1
    return 1


def checkPorts(URL):
    """
    Function that checks if the website has the proper configuration of ports.
    It considers the status of ports: 21, 22, 23, 80, 443, 445, 1433, 1521, 3306, 3389.
    state: ports open is 0,close is:1
    It returns -1 if at most 2 ports are not of the preferred status; 0 if at most 5 ports are not of the preferred status; and 1 otherwise.
    """
    try:
        domain = get_fld(URL)
    except:
        return 1
    suspicious_threshold = 5
    legitimate_threshold = 8
    preferred_status = {
        "21": "close",
        "22": "close",
        "23": "close",
        "80": "open",
        "443": "open",
        "445": "close",
        "1433": "close",
        "1521": "close",
        "3306": "close",
        "3389": "close",
    }
    try:
        domain_str = "nmap %s -p21,22,23,80,443,445,1433,1521,3306,3389" % domain
        return_code, output = subprocess.getstatusoutput(domain_str)
        first = output.find("21/tcp")
        last = output.find("ms-wbt-server")
        out = output[first:last]
        lines = out.splitlines()
        same = 0
        num = 0
        try:
            for key in preferred_status.keys():
                if (lines[num].find(preferred_status.get(key))) > -1:
                    same = same + 1
                num = num + 1

        except:
            num = 0

        if same >= legitimate_threshold:
            return -1
        elif same >= suspicious_threshold:
            return 0
        else:
            return 1
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("error:%s" % e)
    return 1


def checkSSL(URL):
    """
    check if the ssl_certificate is legal
    """
    try:
        domain = get_fld(URL)
    except:
        return 1
    start_date = ""
    expire_date = ""
    cn = ""
    trusted_CAs = [
        "GeoTrust",
        "GoDaddy",
        "Network Solutions",
        "Thawte",
        "Comodo",
        "Doster",
        "Verisign",
        "SSL.com",
        "Secure128",
        "Google",
        "InCommon",
        "Trustico",
        "GlobalSign",
        "SSLRenewals",
        "DigiCert",
        "Symantec",
        "IdenTrust",
        "EnTrust",
        "RapidSSL",
        "Encrypt",
        "Amazon",
        "D-Trust",
        "Starfield",
        "Gandi",
        "Sectigo",
        "Microsoft",
        "WellsFargo",
        "GoGetSSL",
        "QuoVadis",
        "Trusted Secure Certificate Authority",
        "Certum",
        "TrustWave",
        "TeleSec",
        "CyberTrust Japan",
        "DFN-Verein",
        "Actalis",
        "SwissSign",
        "Apple",
        "Affirm",
        "SecureCore",
        "Strato",
        "DonDominio",
        "Globe",
        "Gehirn",
    ]

    duration_threshold = 300
    trusted = False
    try:
        domain_str = "curl -Ivs https://%s --connect-timeout 10" % domain
        return_code, output = subprocess.getstatusoutput(domain_str)
        m = re.search(
            "SSL connection using (.*?)\n.*?start date: (.*?)\n.*?expire date: (.*?)\n.*?issuer: (.*?)\n.*?",
            output,
            re.S,
        )
        if m:
            start_date = m.groups()[1]
            expire_date = m.groups()[2]
            issuer = m.groups()[3]
            start_date = datetime.strptime(start_date, "%b %d %H:%M:%S %Y %Z")
            expire_date = datetime.strptime(expire_date, "%b %d %H:%M:%S %Y %Z")
            length = (expire_date - start_date).days
            # print('length',length)
            dic = {i.split("=")[0]: i.split("=")[1] for i in issuer.split(";")}
            cn = dic[" CN"]
            # check if the ca is trusted
            try:
                issuer = cn.lower()
                for tCA in trusted_CAs:
                    if issuer.find(tCA.lower()) > -1:
                        trusted = True
            except:
                issuer = 0
            if trusted and length > duration_threshold:
                return -1
            else:
                return 1
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("error:%s" % e)
    return 1


# HTML features
def getObjects(HTML):
    """
    get objects from html
    """
    images = HTML.findAll("img")
    links = HTML.findAll("link")
    anchors = HTML.findAll("a")
    sounds = HTML.findAll("sound")
    videos = HTML.findAll("video")
    objects = images + links + anchors + sounds + videos
    # print("[INFO] HTML objects check internal: {}".format(objects))
    return objects


def sameAuthors(element_location, URL):
    """
    Function to determine if two URLs are made by the same authors.
    If the first URL contains one "important" word within the second URL, then it returns True.
    Otherwise it returns False
    """
    element_domain = ((urlparse(element_location)).netloc).lower()
    if len(element_domain) == 0:
        return False
    if URL.count(".") == 1 and URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            domain = URL[:ind]
        else:
            domain = URL
    else:
        domain = ((urlparse(URL)).netloc).lower()
    domain_words = domain.split(".")
    words_to_check = []
    for word in domain_words:
        if len(word) > 3:
            words_to_check.append(word)
    for word in words_to_check:
        if element_domain.find(word) > -1:
            return True
    return False


def isInternal(element_location, base_url):
    """
    Function that determines if the location of an HTML element within a webpage is from a different website (External) or not (Internal).
    An element is "Internal" if its URI adopts relative paths or if its sourced from a webpage from the same (likely) authors as the current one.
    This function returns False if the element is sourced from an external site, and True otherwise.
    """
    if not element_location:
        return True  # Empty is probably internal or broken

    absolute_url = urljoin(base_url, element_location)

    base_netloc = urlparse(base_url).netloc.lower()
    element_netloc = urlparse(absolute_url).netloc.lower()

    if not element_netloc or element_netloc == base_netloc:
        return True
    return sameAuthors(absolute_url, base_url)


def checkObjects(objects, URL):
    """
    Function that checks how many objects embedded in the webpage are from external websites.
    The return value depends on the rate of suspicious anchors, which is compared against 2 thresholds (suspicious (0) and phising (1)).
    """
    suspicious_threshold = 0.15
    phishing_threshold = 0.3
    if len(objects) == 0:
        return -1

    external_objects = []
    suspicious_internal_objects = []
    object_locations = []

    for obj in objects:
        if "src" in obj.attrs:
            object_location = obj["src"]
        elif "href" in obj.attrs:
            object_location = obj["href"]
        else:
            continue

        object_locations.append(object_location)

        # Normalize to absolute URL
        absolute_location = urljoin(URL, object_location)

        if not isInternal(object_location, URL):
            external_objects.append(obj)
        else:
            # Check for suspicious internal objects
            if (
                "?" in object_location or "&" in object_location or
                re.search(r"(ad|advert|update|download|flv|popup)", object_location, re.IGNORECASE)
            ):
                suspicious_internal_objects.append(obj)

    total_objects = len(object_locations)
    suspicious_count = len(external_objects) + len(suspicious_internal_objects)
    suspicious_ratio = suspicious_count / total_objects if total_objects > 0 else 0

    if suspicious_ratio < suspicious_threshold:
        return -1
    elif suspicious_ratio < phishing_threshold:
        return 0
    return 1

def checkMetaScripts(HTML, URL):
    """
    Function that checks the percentage of scripts and metas that share the same domain as the page URL.
    The return value depends on the percentage of external, which is compared against 2 thresholds (suspicious (0) and phishing (1)).
    """
    suspicious_threshold = 0.52
    phishing_threshold = 0.61
    metas = HTML.findAll("meta")
    scripts = HTML.findAll("script")
    links = HTML.findAll("link")
    objects = metas + scripts + links
    if len(objects) == 0:
        return -1  # no embedded objects in html
    external_objects = []
    object_locations = []
    for o in objects:
        object_location = ""
        keys = o.attrs.keys()
        if "src" in keys:
            object_location = o["src"]
            object_locations.append(object_location)
        elif "href" in keys:
            object_location = o["href"]
            object_locations.append(object_location)
        elif "http-equiv" in keys:
            if "content" in keys:
                content = o.attrs["content"]
                content_split = content.split("URL=")
                if len(content_split) > 1:
                    object_location = content_split[1].strip()
                    object_locations.append(object_location)
        if object_location == "":
            continue
        if not (isInternal(object_location, URL)):
            external_objects.append(o)
    if len(object_locations) == 0:
        # print("no linked meta_scripts in html of url: {}".format(URL))
        return -1  # no linked objects in html
    external_objects_rate = len(external_objects) / len(object_locations)
    # return external_objects_rate

    if external_objects_rate < suspicious_threshold:
        return -1
    elif external_objects_rate < phishing_threshold:
        return 0
    return 1


def checkFrequentDomain(objects, URL):  # HTML, URL
    """
    This feature examines all the anchor links in source code of a website and compares the most frequent domain
    with the local domain of a website.If the domains are similar, then the feature is set to 0.
    get the most frequency of ex_domain, if it > the frequency of in_domain, means the most frequent domain is ex_domain
    """
    # get the frequency of external domain
    if len(objects) == 0:
        # print('no objects')
        return -1  # no embedded objects in html
    object_locations = []
    ex_domains = []
    frequency_in = 0
    for o in objects:
        try:
            object_location = o["src"]
            object_locations.append(object_location)
        except:
            try:
                object_location = o["href"]
                object_locations.append(object_location)
            except:
                continue
        if isInternal(object_location, URL):
            frequency_in = frequency_in + 1
        else:
            ex_domain = ((urlparse(object_location)).netloc).lower()
            ex_domains.append(ex_domain)
    # print('object_locations',object_locations)
    ex_domains = [x for x in ex_domains if "w3.org" not in x]
    if len(ex_domains) == 0:
        # print('ex_domain is none')
        return -1
    # print('ex_domains',ex_domains)

    # try:
    #     frequent_ex = max(set(ex_domains), key=ex_domains.count)
    # except:
    #     return -1
    # print('frequent_ex',frequent_ex)

    try:
        frequency_ex = max(ex_domains.count(b) for b in ex_domains if b)
    except:
        frequency_ex = 0
    # print('frequency_ex',frequency_ex)
    # print('frequency_in is',frequency_in)
    # compare frequency of internal or external
    if frequency_in >= frequency_ex:
        return -1
    else:
        return 1


def checkCommonPageRatioinWeb(objects, HTML, URL):
    """
    get the highest frequency of internal or external objects from html
    """
    # get the frequency of external domain
    metas = HTML.findAll("meta")
    scripts = HTML.findAll("script")
    objects = objects + metas + scripts
    if len(objects) == 0:
        return 0  # no embedded objects in html
    object_locations = []
    ex_domains = []
    frequency_in = 0
    for o in objects:
        try:
            object_location = o["src"]
            object_locations.append(object_location)
        except:
            try:
                object_location = o["href"]
                object_locations.append(object_location)
            except:
                continue
        if isInternal(object_location, URL):
            frequency_in = frequency_in + 1
        else:
            ex_domain = ((urlparse(object_location)).netloc).lower()
            ex_domains.append(ex_domain)
    # print('object_locations', object_locations)
    if len(object_locations) == 0:
        # print('no objects')
        return 0  # no embedded url in html
    # print('ex_domains',ex_domains)
    if len(ex_domains) > 0:
        try:
            frequency_ex = max(ex_domains.count(b) for b in ex_domains if b)
        except:
            frequency_ex = 0
    else:
        frequency_ex = 0
    # print('frequency_ex',frequency_ex)
    # print('frequency_in is',frequency_in)
    if frequency_in >= frequency_ex:
        most_frequent = frequency_in
    else:
        most_frequent = frequency_ex
    total = len(object_locations)
    ratio = most_frequent / total
    return float(format(ratio, ".3f"))


def checkCommonPageRatioinFooter(HTML, URL):
    """
    get the highest frequency from footer

    """
    foot = HTML.footer
    # print('foot',foot)
    if foot is None:
        return 0
    images = foot.findAll("img")
    links = foot.findAll("link")
    anchors = foot.findAll("a")
    sounds = foot.findAll("sound")
    videos = foot.findAll("video")
    metas = foot.findAll("meta")
    li = foot.findAll("li")
    scripts = foot.findAll("script")
    objects = images + links + anchors + sounds + videos + metas + scripts + li
    if len(objects) == 0:
        # print('no objects')
        return 0  # no embedded objects in html
    object_locations = []
    ex_domains = []
    frequency_in = 0
    for o in objects:
        try:
            object_location = o["src"]
            object_locations.append(object_location)
        except:
            try:
                object_location = o["href"]
                object_locations.append(object_location)
            except:
                continue
        if isInternal(object_location, URL):
            frequency_in = frequency_in + 1
        else:
            ex_domain = ((urlparse(object_location)).netloc).lower()
            ex_domains.append(ex_domain)
    # print('object_locations', object_locations)
    if len(object_locations) == 0:
        # print('no objects')
        return 0  # no embedded url in html
    # print('ex_domains', ex_domains)
    if len(ex_domains) > 0:
        try:
            frequency_ex = max(ex_domains.count(b) for b in ex_domains if b)
        except:
            frequency_ex = 0
    else:
        frequency_ex = 0

    # print('frequency_ex', frequency_ex)
    # print('frequency_in is', frequency_in)
    if frequency_in >= frequency_ex:
        most_frequent = frequency_in
    else:
        most_frequent = frequency_ex
    # print('most_frequent',most_frequent)
    total = len(object_locations)
    # print('total',total)
    ratio = most_frequent / total
    return float(format(ratio, ".3f"))


def checkSFH(HTML, URL):
    """
    Function that checks how many forms are suspicious.
    The return value depends on the rate of suspicious FORMS, which is compared against 2 thresholds (suspicious (0) and phising (1)).
    """
    suspicious_threshold = 0.5
    phishing_threshold = 0.75
    forms = HTML.findAll("form")
    if len(forms) == 0:
        return -1  # no forms in html
    suspicious_forms = []
    for form in forms:
        if "action" in form:
            form_location = form["action"]
            if not (isInternal(form_location, URL)):
                suspicious_forms.append(form)
            elif form_location == "about:blank":
                suspicious_forms.append(form)
            elif form_location == "":
                suspicious_forms.append(form)
    suspicious_forms_rate = len(suspicious_forms) / len(forms)

    if suspicious_forms_rate < suspicious_threshold:
        return -1
    elif suspicious_forms_rate < phishing_threshold:
        return 0
    return 1


def checkPopUp(HTML):
    """
    Function that checks if the HTML contains code that triggers a popup window with input text fields.
    These elements are introduced with the "prompt()" code. Other popup windows can be introduced with the code "window.open()".
    This function returns 1 if the HTML contains popup windows with text fields; 0 if it contains any popup window; and -1 if no popup windows are found.
    """
    if str(HTML).find("prompt(") >= 0:
        return 1
    elif str(HTML).find("window.open(") >= 0:
        return 0
    return -1


def checkRightClick(HTML):
    """
    Function that inspects the provided HTML to determine if the CONTEXTMENU has been disabled (which is the equivalent of disabling the mouse right click)
    This can be performed in several ways.
    It returns 1 if the contextmenu is disabled, and -1 otherwise.
    """
    # contextmenu_disabler_JS = "preventDefault()"
    contextmenu_disabler_html = 'oncontextmenu="return false;"'  # ;

    if str(HTML).find(contextmenu_disabler_html) >= 0:
        # print("found oncontextmenu")
        return 1
    # elif HTML.find(contextmenu_disabler_JS) > =0:
    #     print("found preventDefault")
    #     return 1
    return -1


def checkDomainwithCopyright(HTML, URL):
    """
    Function that checks if the website's domain in the copyright
    """
    try:
        res = get_tld(URL, as_object=True)
    except:
        return 1
    domain = res.domain
    # print('domain is:',domain)
    symbol = "\N{COPYRIGHT SIGN}".encode("utf-8")
    symbol = symbol.decode("utf-8")
    pattern = r"" + symbol
    if len(HTML.findAll(text=re.compile(pattern))) < 1:
        # print('no copyright')
        return 0  # no copyright
    for tag in HTML.findAll(text=re.compile(pattern)):
        copyrightTexts = tag.parent.text
        # print('copytest is', copyrightTexts)
        if copyrightTexts.find(domain) > -1:
            # print('domain in copyrighttexts')
            return -1
    return 1


def nullLinksinWeb(HTML, URL):
    """
    Function that checks how many suspicious anchors are contained in a website.
    The return value depends on the number of  suspicious anchors
    """
    anchors = HTML.findAll("a")
    if len(anchors) == 0:
        return 0  # no anchors in html
    suspicious_anchors = []
    for a in anchors:
        try:
            anchor_location = a["href"]
        except:
            continue
        if anchor_location == "#":
            suspicious_anchors.append(a)
        elif anchor_location == "#content":
            suspicious_anchors.append(a)
        elif anchor_location == "#skip":
            suspicious_anchors.append(a)
        elif anchor_location == "JavaScript ::void(0)":
            suspicious_anchors.append(a)
        elif isInternal(anchor_location, URL):
            suspicious_anchors.append(a)

    suspicious_anchors_rate = len(suspicious_anchors) / len(anchors)

    # print("suspicious anchors: {}".format(suspicious_anchors))
    # print("suspicious anchors rate: {}".format(suspicious_anchors_rate))

    return float(format(suspicious_anchors_rate, ".2f"))


def nullLinksinFooter(HTML, URL):
    """
    Function that checks how many suspicious anchors are contained in the footer.
    The return value depends on the number of  suspicious anchors

    """
    foot = HTML.footer
    # print('foot',foot)
    suspicious_anchors = []
    if foot is None:
        return 0
    anchors = foot.findAll("a")
    if len(anchors) == 0:
        return 0

    for a in anchors:
        try:
            anchor_location = a["href"]
        except:
            continue
        if anchor_location == "#":
            suspicious_anchors.append(a)
        elif anchor_location == "#content":
            suspicious_anchors.append(a)
        elif anchor_location == "#skip":
            suspicious_anchors.append(a)
        elif anchor_location == "JavaScript ::void(0)":
            suspicious_anchors.append(a)
        # elif((isInternal(anchor_location, URL))):
        # suspicious_anchors.append(a)
    suspicious_anchors_rate = len(suspicious_anchors) / len(anchors)
    """
    print('foot',foot)
    print('suspicious anchors',suspicious_anchors)
    print('len of suspicious anchors',len(suspicious_anchors))
    print('len of anchors',len(anchors))
    print('suspicious_anchors_rate',suspicious_anchors_rate)
    """
    return float(format(suspicious_anchors_rate, ".2f"))


def checkBrokenLink(HTML, URL):
    """
    This feature extracts ratio of Not found links to the total number of links in a website. In legitimate sites, when all the links are
    connected either they return 200 Ok HTTP status code indicating server has accepted the request or sends 404
    status code indicating page not found in that server.broken_anchor/total_anchor
    """
    # get the frequency of external domain
    images = HTML.findAll("img")
    links = HTML.findAll("link")
    anchors = HTML.findAll("a")
    sounds = HTML.findAll("sound")
    videos = HTML.findAll("video")
    metas = HTML.findAll("meta")
    scripts = HTML.findAll("script")
    objects = images + links + anchors + sounds + videos + metas + scripts
    broken_link = 0
    if len(objects) == 0:
        return 0  # no embedded objects in html
    object_locations = []
    for o in objects:
        try:
            object_location = o["src"]

            if not (isInternal(object_location, URL)):
                object_locations.append(object_location)

        except:
            try:
                object_location = o["href"]

                if not (isInternal(object_location, URL)):
                    object_locations.append(object_location)

            except:
                continue

    if len(object_locations) == 0:
        return 0  # no embedded url in html

    for obj in object_locations:
        try:
            resp = urllib.request.urlopen(obj, timeout=2)
            status_code = resp.getcode()
            if status_code >= 400:
                broken_link = broken_link + 1
        except:
            broken_link = broken_link + 1
    broken_link_rate = broken_link / len(object_locations)
    return float(format(broken_link_rate, ".2f"))


def checkLoginForm(HTML, URL):
    """
    function that checks if the form sent to a suspicious website(external or null)
    """
    # get the frequency of external domain
    forms = HTML.findAll("form")
    empty = [
        "",
        "#",
        "#nothing",
        "#doesnotexist",
        "#null",
        "#void",
        "#whatever",
        "#content",
        "javascript::void(0)",
        "javascript::void(0);",
        "javascript::;",
        "javascript",
    ]
    for obj in forms:
        if "action" in obj.attrs:
            if obj["action"] in empty or not (isInternal(obj["action"], URL)):
                return 1
    return -1


def checkHiddenInfo_div(HTML):
    """
    Some special codes in HTML can prevent the content from displaying or restricting the function of a tag,
    which may be used by phishing webpages. These special codes work on specific tags,
    1.<div>:<div style="visibility:hidden",<div style="display:none">
    2.<button disabled='disabled'>
    3.<input type=hidden><input disabled='diabled'><input value='hello'> fills in some irrelevant info in the input box
    """
    divs = HTML.findAll("div")
    for div in divs:
        if "style" in div.attrs and (
            "visibility:hidden" in div["style"] or "display:none" in div["style"]
        ):
            return 1
    return -1


def checkHiddenInfo_button(HTML):
    buttons = HTML.findAll("button")
    for button in buttons:
        if "disabled" in button.attrs and button["disabled"] in ["", "disabled"]:
            return 1
    return -1


def checkHiddenInfo_input(HTML):
    inputs = HTML.findAll("input")

    for inp in inputs:
        if ("type" in inp.attrs and inp["type"] == "hidden") or "disabled" in inp.attrs:
            return 1
    return -1


def checkTitleUrlBrand(HTML, URL):
    """
    function that checks if the title includes the website's domain
    """

    try:
        domain_brand = (get_tld(URL, as_object=True)).domain
    except:
        return 1

    try:
        title = HTML.find("title").get_text()

        if len(title) < 2:  # if title is null or space->suspicious
            return 0
        elif title.find(domain_brand) > -1:
            return -1
        else:
            return 1
    except:
        return 0


def checkIFrame(HTML):
    """
    function that checks the hidden iframe

    """
    iframes = HTML.find_all("iframe")
    for iframe in iframes:
        try:
            if (
                (iframe["style"].find("display: none") > -1)
                or (iframe["style"].find("border: 0") > -1)
                or (iframe["style"].find("visibility: hidden;") > -1)
                or (iframe["frameborder"].find("0") > -1)
            ):
                return 1
        except:
            continue
    return -1


def checkFavicon(HTML, URL):
    """
    Function that checks if the Favicon of the website comes from an external source.
    It returns 1 if it's from an external source; 0 if it does not have a Favicon. And -1 if the Favicon is internal.
    """
    favicon = HTML.find(rel="shortcut icon")

    if not favicon:
        favicon = HTML.find(rel="icon")
    if favicon:
        if "href" in favicon.attrs:
            if isInternal(favicon["href"], URL):
                return -1
            else:
                return 1

    return 0


def checkStatusBar(HTML):
    """
    Function that inspects the provided HTML to determine if it changes the text of the statusbar.
    It returns 1 if statusbar modifications are detected, and -1 otherwise.
    """
    status_bar_modification = "window.status"
    if str(HTML).find(status_bar_modification) >= 0:
        return 1
    return -1


def checkCSS(HTML, URL):
    """
    Function that checks if the CSS of the website comes from an external source.
    It returns 1 if it's from an external source; and -1 otherwise.
    """
    css = HTML.find(rel="stylesheet")
    if css is not None and "href" in css.attrs and not (isInternal(css["href"], URL)):
        return 1
    return -1


def checkAnchors(HTML, URL):
    """
    Function that checks how many suspicious anchors are contained in a website.
    The return value depends on the rate of suspicious anchors, which is compared against 2 thresholds (suspicious (0) and phising (-1)).
    """
    suspicious_threshold = 0.32
    phishing_threshold = 0.505
    anchors = HTML.findAll("a")
    if len(anchors) == 0:
        return -1  # no anchors in html
    suspicious_anchors = []
    for a in anchors:
        try:
            anchor_location = a["href"]
        except:
            continue
        if anchor_location == "#":
            suspicious_anchors.append(a)
        elif anchor_location == "#content":
            suspicious_anchors.append(a)
        elif anchor_location == "#skip":
            suspicious_anchors.append(a)
        elif anchor_location == "JavaScript ::void(0)":
            suspicious_anchors.append(a)
        elif not (isInternal(anchor_location, URL)):
            suspicious_anchors.append(a)
    suspicious_anchors_rate = len(suspicious_anchors) / len(anchors)
    if suspicious_anchors_rate < suspicious_threshold:
        return -1
    elif suspicious_anchors_rate < phishing_threshold:
        return 0
    return 1


# extract html features
def extract_features_html(HTML, URL):
    h_features = {}

    objects = time_feature("HTML_getObjects", getObjects, HTML)

    h_features["HTML_Objects"] = time_feature("HTML_Objects", checkObjects, objects, URL)
    h_features["HTML_metaScripts"] = time_feature("HTML_metaScripts", checkMetaScripts, HTML, URL)
    h_features["HTML_FrequentDomain"] = time_feature("HTML_FrequentDomain", checkFrequentDomain, objects, URL)
    h_features["HTML_Commonpage"] = time_feature("HTML_Commonpage", checkCommonPageRatioinWeb, objects, HTML, URL)
    h_features["HTML_CommonPageRatioinFooter"] = time_feature("HTML_CommonPageRatioinFooter", checkCommonPageRatioinFooter, HTML, URL)
    h_features["HTML_SFH"] = time_feature("HTML_SFH", checkSFH, HTML, URL)
    h_features["HTML_popUp"] = time_feature("HTML_popUp", checkPopUp, HTML)
    h_features["HTML_RightClick"] = time_feature("HTML_RightClick", checkRightClick, HTML)
    h_features["HTML_DomainwithCopyright"] = time_feature("HTML_DomainwithCopyright", checkDomainwithCopyright, HTML, URL)
    h_features["HTML_nullLinksinWeb"] = time_feature("HTML_nullLinksinWeb", nullLinksinWeb, HTML, URL)
    h_features["HTML_nullLinksinFooter"] = time_feature("HTML_nullLinksinFooter", nullLinksinFooter, HTML, URL)
    h_features["HTML_BrokenLink"] = time_feature("HTML_BrokenLink", checkBrokenLink, HTML, URL)
    h_features["HTML_LoginForm"] = time_feature("HTML_LoginForm", checkLoginForm, HTML, URL)
    h_features["HTML_HiddenInfo_div"] = time_feature("HTML_HiddenInfo_div", checkHiddenInfo_div, HTML)
    h_features["HTML_HiddenInfo_button"] = time_feature("HTML_HiddenInfo_button", checkHiddenInfo_button, HTML)
    h_features["HTML_HiddenInfo_input"] = time_feature("HTML_HiddenInfo_input", checkHiddenInfo_input, HTML)
    h_features["HTML_TitleUrlBrand"] = time_feature("HTML_TitleUrlBrand", checkTitleUrlBrand, HTML, URL)
    h_features["HTML_IFrame"] = time_feature("HTML_IFrame", checkIFrame, HTML)
    h_features["HTML_favicon"] = time_feature("HTML_favicon", checkFavicon, HTML, URL)
    h_features["HTML_statusBarMod"] = time_feature("HTML_statusBarMod", checkStatusBar, HTML)
    h_features["HTML_css"] = time_feature("HTML_css", checkCSS, HTML, URL)
    h_features["HTML_anchors"] = time_feature("HTML_anchors", checkAnchors, HTML, URL)

    return h_features

# extract URL features
def extract_features_url(URL):
    """
    Function that takes a JSON object as input, and computes the URL features.
    These features include: URL_IP, URL_redirect, URL_long, URL_shortener, URL_subdomains, URL_at, URL_fakeHTTPS, URL_dash, URL_dataURI
    The output is a JSON object representing the values of each of the abovementioned features for the input object.
    """

    u_features = {}

    u_features["URL_length"] = time_feature("URL_length", checkLength, URL)
    u_features["URL_IP"] = time_feature("URL_IP", checkIP, URL)
    u_features["URL_redirect"] = time_feature("URL_redirect", checkRedirect, URL)
    u_features["URL_shortener"] = time_feature("URL_shortener", checkShortener, URL)
    u_features["URL_subdomains"] = time_feature("URL_subdomains", checkSubdomains, URL)
    u_features["URL_at"] = time_feature("URL_at", checkAt, URL)
    u_features["URL_fakeHTTPS"] = time_feature("URL_fakeHTTPS", checkFakeHTTPS, URL)
    u_features["URL_dash"] = time_feature("URL_dash", checkDash, URL)
    u_features["URL_dataURI"] = time_feature("URL_dataURI", checkDataURI, URL)
    u_features["URL_numberofCommonTerms"] = time_feature("URL_numberofCommonTerms", checkNumberofCommonTerms, URL)
    u_features["URL_checkNumerical"] = time_feature("URL_checkNumerical", checkNumerical, URL)
    u_features["URL_checkPathExtend"] = time_feature("URL_checkPathExtend", checkPathExtend, URL)
    u_features["URL_checkPunycode"] = time_feature("URL_checkPunycode", checkPunycode, URL)
    u_features["URL_checkSensitiveWord"] = time_feature("URL_checkSensitiveWord", checkSensitiveWord, URL)
    u_features["URL_checkTLDinPath"] = time_feature("URL_checkTLDinPath", checkTLDinPath, URL)
    u_features["URL_checkTLDinSub"] = time_feature("URL_checkTLDinSub", checkTLDinSub, URL)
    u_features["URL_totalWordUrl"] = time_feature("URL_totalWordUrl", totalWordUrl, URL)
    u_features["URL_shortestWordUrl"] = time_feature("URL_shortestWordUrl", shortestWordUrl, URL)
    u_features["URL_shortestWordHost"] = time_feature("URL_shortestWordHost", shortestWordHost, URL)
    u_features["URL_shortestWordPath"] = time_feature("URL_shortestWordPath", shortestWordPath, URL)
    u_features["URL_longestWordUrl"] = time_feature("URL_longestWordUrl", longestWordUrl, URL)
    u_features["URL_longestWordHost"] = time_feature("URL_longestWordHost", longestWordHost, URL)
    u_features["URL_longestWordPath"] = time_feature("URL_longestWordPath", longestWordPath, URL)
    u_features["URL_averageWordUrl"] = time_feature("URL_averageWordUrl", averageWordUrl, URL)
    u_features["URL_averageWordHost"] = time_feature("URL_averageWordHost", averageWordHost, URL)
    u_features["URL_averageWordPath"] = time_feature("URL_averageWordPath", averageWordPath, URL)
    u_features["URL_checkStatisticRe"] = time_feature("URL_checkStatisticRe", checkStatisticRe, URL)

    return u_features


def extract_features_rep(URL):
    r_features = {}

    # WHOIS is expensive → time it
    who = time_feature("REP_getWhois", getWhois, URL)

    #r_features["REP_pageRank"] = time_feature("REP_pageRank", checkPR, URL)
    r_features["REP_DNS"] = time_feature("REP_DNS", checkDNS, who)
    r_features["REP_registrationLen"] = time_feature("REP_registrationLen", checkRegistrationLen, who)
    r_features["REP_Age"] = time_feature("REP_Age", checkAge, URL, who)
    r_features["REP_abnormal"] = time_feature("REP_abnormal", checkAbnormal, who, URL)
    r_features["REP_ports"] = time_feature("REP_ports", checkPorts, URL)
    r_features["REP_SSL"] = time_feature("REP_SSL", checkSSL, URL)

    return r_features

def extract_features_specific(HTML, URL):
    objects = time_feature("HTML_getObjects", getObjects, HTML)
    features = {}

    features["URL_longestWordPath"] = time_feature("URL_longestWordPath", longestWordPath, URL)
    features["URL_totalWordUrl"] = time_feature("URL_totalWordUrl", totalWordUrl, URL)
    features["URL_averageWordPath"] = time_feature("URL_averageWordPath", averageWordPath, URL)
    features["URL_longestWordUrl"] = time_feature("URL_longestWordUrl", longestWordUrl, URL)
    features["URL_shortestWordPath"] = time_feature("URL_shortestWordPath", shortestWordPath, URL)
    features["URL_subdomains"] = time_feature("URL_subdomains", checkSubdomains, URL)
    features["HTML_HiddenInfo_input"] = time_feature("HTML_HiddenInfo_input", checkHiddenInfo_input, HTML)
    features["URL_dash"] = time_feature("URL_dash", checkDash, URL)
    features["HTML_IFrame"] = time_feature("HTML_IFrame", checkIFrame, HTML)
    features["HTML_anchors"] = time_feature("HTML_anchors", checkAnchors, HTML, URL)
    features["URL_length"] = time_feature("URL_length", checkLength, URL)
    features["URL_checkNumerical"] = time_feature("URL_checkNumerical", checkNumerical, URL)
    features["HTML_MetaScripts"] = time_feature("HTML_metaScripts", checkMetaScripts, HTML, URL)
    features["HTML_FrequentDomain"] = time_feature("HTML_FrequentDomain", checkFrequentDomain, objects, URL)
    features["HTML_css"] = time_feature("HTML_css", checkCSS, HTML, URL)
    features["URL_longestWordHost"] = time_feature("URL_longestWordHost", longestWordHost, URL)
    # features["REP_pageRank"] = time_feature("REP_pageRank", checkPR, URL)
    return features




def extract_features_phishing(HTML, URL, feat_type="all"):
    assert feat_type in {"all", "url", "html","compressed"}

    if feat_type in ["all", "html"]:
        html_features = extract_features_html(HTML, URL)
    if feat_type in ["all", "url"]:
        url_features = extract_features_url(URL)
        rep_features = extract_features_rep(URL)
    if feat_type == "compressed":
        features_compressed = extract_features_specific(HTML,URL)

    features_dict = {}
    if feat_type == "html":
        features = list(html_features.values())
        features_dict.update(html_features)
    elif feat_type == "url":
        features = list(url_features.values()) + list(rep_features.values())
        features_dict.update(url_features)
        features_dict.update(rep_features)
    elif feat_type == "compressed":
        features = list(features_compressed.values())
        features_dict.update(features_compressed)
    else:
        features = (
            list(url_features.values())
            + list(html_features.values())
            + list(rep_features.values())
        )
        features_dict.update(url_features)
        features_dict.update(html_features)
        features_dict.update(rep_features)

    # print(features_dict)
    # print(np.array(features).astype("float"))

    return np.array(features).astype("float")


# utilities
def build_phishing_test_data_info(main_filepath, test_samples_path, out_file_path):
    phish_test_set = joblib.load(test_samples_path)
    test_samples_idx = phish_test_set.index.to_list()

    with open(main_filepath, encoding="utf-8") as data_file:
        data = json.loads(data_file.read())

    samples_info = []
    for idx in test_samples_idx:
        sample = data[idx]
        samples_info.append({"id": sample["id"], "url": sample["url"]})

    with open(out_file_path, "wb") as fp:
        pickle.dump(samples_info, fp)


