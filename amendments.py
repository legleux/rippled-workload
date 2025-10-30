from dataclasses import dataclass
import urllib.request
import json
from pathlib import Path
from enum import IntEnum
from urllib.parse import urlparse

class Network(IntEnum):
    MAIN = 0
    TEST = 1
    DEV  = 2

    @property
    def id(self):
        return self.value

network_rpc_url = {
    0: [
        "https://s1.ripple.com:51234",
        "https://s2.ripple.com:51234",
    ],
    1: [
        "https://s.altnet.rippletest.net:51234/",
        "https://clio.altnet.rippletest.net:51234/",
    ],
    2: ["https://s.devnet.rippletest.net:51234"],
}

DEFAULT_NETWORK = Network.DEV

use_default = False
if use_default:
    from ledger_tools import data_dir
    DEFAULT_AMENDMENT_LIST = data_dir / "amendment_list_dev_20250907.json"

# https://xrpl.org/resources/known-amendments#dynamicnft
@dataclass(frozen=True, slots=True)
class Amendment:
    index: str
    name: str
    link: str
    enabled: bool
    obsolete: bool = False

    def __str__(self):
        enabled = "Enabled" if self.enabled else "Disabled"
        return f"{self.name} {enabled}"

def _get_amendments_from_file(amendments_file: str | None = None) -> list[Amendment]:
    """Return list of amendments from file as rippled feature list"""
    if amendments_file is not None:
        features_file = Path(amendments_file)
    else:
        features_file = DEFAULT_AMENDMENT_LIST
    return json.loads(features_file.resolve().read_text())

def _get_amendments(source: str | None = None)-> list[Amendment]:
    if isinstance(source, str) or source is None:
        return _get_amendments_from_file(source)
    else:
        return _get_amendments_from_net(source)

def _get_amendments_from_net(network: Network | None, timeout: int = 3) -> list[Amendment]:
    """Get the amendments enabled on the `network` via `rippled`'s `feature` command."""
    network = network or DEFAULT_NETWORK
    urls = network_rpc_url[network]
    payload = {"method": "feature"}
    data = json.dumps(payload).encode("utf-8")
    try:
        for url in urls:
            response = urllib.request.urlopen(url, data=data, timeout=timeout)
            res = json.loads(response.read())
            amend = res["result"]["features"]
            return amend
    except Exception as e:
        pass

def get_amendments(source: str | None = None) -> list[Amendment]:
    """
    Accepts: source to read amendments from.
    Returns: list[Amendment]
    """
    prefix="https://xrpl.org/resources/known-amendments#"
    a = _get_amendments(source)
    ams: list[Amendment] = []
    for am_hash, info in a.items():
        ams.append(
            Amendment(
                index=am_hash,
                name=(name := info.get("name", am_hash)),
                link=prefix + name.lower(),
                enabled=bool(info.get("enabled", False)),
                obsolete=bool(info.get("obsolete", False)),
            )
        )
    return ams

def get_enabled_amendment_hashes(source: str | None = None) -> list[str]:
    a = get_amendments(source)
    return _enabled_amendment_hashes(a)

def _enabled_amendment_hashes(amendments: list[Amendment]) -> list[str]:
    return [a.index for a in amendments if a.enabled]

def get_devnet_amendments():
    a = get_amendments(Network.DEV)
    return a

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network", help="Network to query amendments")
    # parser.add_argument("-u", "--url", help="rippled node to query for enabled amendments")
    parser.add_argument("-e", "--enabled", action='store_true')
    # parser.add_argument('--supported', action='store_true')
    args = parser.parse_args()
    print(args)

    for a in get_devnet_amendments():
        print(a)
