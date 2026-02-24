# /// script
# requires-python = ">=3.13"
# ///
import json
import urllib.request
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
import hashlib

amendments = [
    "AMM",
    "AMMClawback",
    "Batch",
    "CheckCashMakesTrustLine",
    "Checks",
    "Clawback",
    "Credentials",
    "CryptoConditions",
    "CryptoConditionsSuite",
    "DeepFreeze",
    "DeletableAccounts",
    "DepositAuth",
    "DepositPreauth",
    "DID",
    "DisallowIncoming",
    "DynamicNFT",
    "EnforceInvariants",
    "Escrow",
    "ExpandedSignerList",
    "FeeEscalation",
    "Flow",
    "FlowCross",
    "FlowSortStrands",
    "HardenedValidations",
    "ImmediateOfferKilled",
    "MPTokensV1",
    "MultiSign",
    "MultiSignReserve",
    "NegativeUNL",
    "NFTokenMintOffer",
    "NonFungibleTokensV1_1",
    "NonFungibleTokensV1",
    "PayChan",
    "PermissionedDEX",
    "PermissionedDomains",
    "PriceOracle",
    "RequireFullyCanonicalSig",
    "SortedDirectories",
    "TicketBatch",
    "TickSize",
    "TokenEscrow",
    "TrustSetAuth",
    "XChainBridge",
    "XRPFees",
    ### Fix Amendments ###
    "fix1201",
    "fix1368",
    "fix1373",
    "fix1512",
    "fix1513",
    "fix1515",
    "fix1523",
    "fix1528",
    "fix1543",
    "fix1571",
    "fix1578",
    "fix1623",
    "fix1781",
    "fixAmendmentMajorityCalc",
    "fixAMMClawbackRounding",
    "fixAMMOverflowOffer",
    "fixAMMv1_1",
    "fixAMMv1_2",
    "fixAMMv1_3",
    "fixCheckThreading",
    "fixDirectoryLimit",
    "fixDisallowIncomingV1",
    "fixEmptyDID",
    "fixEnforceNFTokenTrustline",
    "fixEnforceNFTokenTrustlineV2",
    "fixFillOrKill",
    "fixFrozenLPTokenTransfer",
    "fixIncludeKeyletFields",
    "fixInnerObjTemplate",
    "fixInnerObjTemplate2",
    "fixInvalidTxFlags",
    "fixMasterKeyAsRegularKey",
    "fixMPTDeliveredAmount",
    "fixNFTokenDirV1",
    "fixNFTokenNegOffer",
    "fixNFTokenPageLinks",
    "fixNFTokenRemint",
    "fixNFTokenReserve",
    "fixNonFungibleTokensV1_2",
    "fixPayChanCancelAfter",
    "fixPayChanRecipientOwnerDir",
    "fixPreviousTxnID",
    "fixPriceOracleOrder",
    "fixQualityUpperBound",
    "fixReducedOffersV1",
    "fixReducedOffersV2",
    "fixRemoveNFTokenAutoTrustLine",
    "fixRmSmallIncreasedQOffers",
    "fixSTAmountCanonicalize",
    "fixTakerDryOfferRemoval",
    "fixTokenEscrowV1",
    "fixTrustLinesToSelf",
    "fixUniversalNumber",
    "fixXChainRewardRounding",
    ]

def sha512half(data):
    return hashlib.sha512(data.encode("utf-8")).digest()[:32].hex().upper()

class Network(IntEnum):
    MAIN = 0
    TEST = 1
    DEV = 2

    @property
    def id(self):
        return self.value

    def __str__(self):
        return f"{self.name.lower()}net".title()


network_rpc_url = {
    0: [
        "https://s1.ripple.com:51234",
        "https://s2.ripple.com:51234",
        "https://r.ripple.com:51234",
    ],
    1: [
        "https://s.altnet.rippletest.net:51234/",
        "https://s1.altnet.rippletest.net:51234/",
        "https://s2.altnet.rippletest.net:51234/",
        "https://r.altnet.rippletest.net:51234/",
        "https://clio.altnet.rippletest.net:51234/",
    ],
    2: ["https://s.devnet.rippletest.net:51234"],
}

DEFAULT_NETWORK = Network.DEV
DEFAULT_AMENDMENT_LIST = "amendment_list_dev_20251118.json"

@dataclass(slots=True)
class Amendment:
    """
    XRPL amendment metadata with XRPL-specific states.

    Fields
    ------
    index : str
        Amendment identifier. Prefer the 256-bit feature ID hex if available.
    name : str
        Canonical amendment name as used by rippled (e.g., "Checks", "fixNFTokenDirV1").
    link : str
        Reference URL in XRPL docs. (e.g., [Dynamic NFTs](https://xrpl.org/resources/known-amendments#dynamicnft)
    enabled : bool
        True if the amendment is active on the XRPL. An amendment becomes enabled
        after it holds >80% validator support for two weeks and then activates,
        after which the rule change is permanent.
    obsolete : bool
        True if the amendment is marked obsolete in source.
    """

    index: str
    name: str
    link: str
    enabled: bool
    obsolete: bool = False # Maybe not default to false

    def __str__(self):
        return f"{self.name} {("Enabled" if self.enabled else "Disabled")}"

    @classmethod
    def name_to_hash(cls, name: str) -> str:
        return sha512half(name)


def _get_amendments_from_file(amendments_file: Path | None = None) -> list[Amendment]:
    """Return list of amendments from file as rippled feature list.

    Args:
        amendments_file (str | None, optional): _description_. Defaults to None.

    Returns:
        list[Amendment]: _description_

    """
    features_file = Path(amendments_file) if amendments_file is not None else DEFAULT_AMENDMENT_LIST
    try:
        return json.loads(features_file.resolve().read_text())
    except Exception as e:
        pass  # probably file not found...

def _get_amendments_from_url(url: str, timeout: int = 3) -> list[Amendment]:
    # network = network or DEFAULT_NETWORK
    # urls = network_rpc_url[network]
    payload = {"method": "feature"}
    data = json.dumps(payload).encode("utf-8")
    try:
        response = urllib.request.urlopen(url, data=data, timeout=timeout)
        res = json.loads(response.read())
        amend = res["result"]["features"]
        # return amend
    except urllib.error.URLError as e:
        msg = f"Couldn't query rippled at {url}"
        raise SystemExit(f"{msg}: {e.reason or e}")
    except KeyError as e:
        print(f"Couldn't query {url} for Amendments!")
        raise SystemExit(f"Response had no key: {e}")
    return amend

def _get_amendments_from_net(network: Network) -> tuple[str, list[Amendment]]:
    """Get the amendments enabled on the `network` via `rippled`'s `feature` command."""
    # BUG: rippled `feature` is _not_ on custom networks!
    urls = network_rpc_url[network]
    for url in urls:
        try:
            a = _get_amendments_from_url(url)
            return url, a
        except Exception:
            continue

    raise RuntimeError(f"failed to fetch amendments for {network}")

def get_amendments(source: Path | str | Network) -> tuple[str, list[Amendment]]:
    prefix="https://xrpl.org/resources/known-amendments#"
    if isinstance(source, Path):
        a = _get_amendments_from_file(source)
    elif isinstance(source, str):
        a = _get_amendments_from_url(source)
        # for i in a.items():
        #     print(i)
        # import sys
        # sys.exit(0)
    else:
        source, a = _get_amendments_from_net(source)

    return str(source), [Amendment(
                index=am_hash,
                name=(name := info.get("name", am_hash)),
                link=prefix + name.lower(),
                enabled=bool(info.get("enabled", False)),
                obsolete=bool(info.get("obsolete", False)),
                ) for am_hash, info in a.items()
            ]

# def get_disabled_amendments(net: Network) -> list[Amendment]:
#     return filter(lambda a: a.enabled, get_amendments(net))
    # disabled = filter(lambda a: a.enabled, get_amendments(amendments))


AmendmentEnum = StrEnum('Amendment', {sha512half(a):a for a in amendments})

# def get_enabled_amendment_hashes(source: Path | Network) -> list[str]:
#     return [a.index for a in get_amendments(source) if a.enabled]


def print_amendments(amendments: list[Amendment]) -> None:
    def _status(a: Amendment) -> tuple[str, int]:
        if a.obsolete:
            return "obsolete", 2
        if a.enabled:
            return "enabled", 0
        return "disabled", 1

    amendments = sorted(amendments, key=lambda a: _status(a)[1])

    for a in amendments:
        status, _ = _status(a)
        print(f"{a.name[:31]:31}  {a.index:8}  {status:9}  {a.link}")
    print()

def parse_ledger(ledger: str):
    ledger_file = Path(ledger).resolve().read_text()
    ledger_json = json.loads(ledger_file)
    account_state = ledger_json['ledger']['accountState']
    amendment_hashes = next((d["Amendments"] for d in account_state if "Amendments" in d))
    return {h:AmendmentEnum[h].value for h in amendment_hashes}

def get_amendments_from_ledger(ledger: str):
    prefix="https://xrpl.org/resources/known-amendments#"
    a = parse_ledger(ledger)
    return [Amendment(
                index=am_hash,
                name=name,
                link=prefix + name.lower(),
                enabled=True,
                ) for am_hash, name in a.items()
            ]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=False)
    status = parser.add_mutually_exclusive_group(required=False)
    default_network="mainnet"
    group.add_argument("network",
                       nargs="?",
                       choices=["mainnet", "testnet", "devnet"],
                       default=default_network,
                       help=f"Network to use (default: {default_network})",
                       )
    group.add_argument("-l", "--ledger", type=Path)
    group.add_argument("-u", "--url", help="rippled node to query for enabled amendments")
    status.add_argument("-d", "--disabled", action='store_true')
    status.add_argument("-e", "--enabled", action='store_true')
    parser.add_argument("-p", "--plain", action='store_true')
    parser.add_argument("-n", "--names-only", action='store_true')
    # parser.add_argument("-j", "--json", action='store_true')
    # parser.add_argument('--supported', action='store_true')
    args = parser.parse_args()

    if args.url:
        net = "Custom"
        source = args.url
        src, amendments = get_amendments(source)
    elif args.ledger:
        amendments = get_amendments_from_ledger(args.ledger)
        pass
    elif args.network:
        match net := args.network:
            case _ if net.startswith(("main", "live", "m")):
                net = Network.MAIN
            case _ if net.startswith(("test", "alt", "t")):
                net = Network.TEST
            case _ if net.startswith(("dev", "d")):
                net = Network.DEV
        src, amendments = get_amendments(net)


    if args.disabled:
        amd = [a for a in amendments if not a.enabled]
        disabled = True
        amendments = list(filter(lambda a: not a.enabled, amendments))

    if args.enabled:
        amd = [a for a in amendments if not a.enabled]
        disabled = True
        amendments = list(filter(lambda a: a.enabled, amendments))

    if args.names_only:
        for a in amd:
            print(a.name)

    if args.plain:
        for a in amd:
            print(a)

    print_amendments(amendments)

    # TODO: intersection of networks
