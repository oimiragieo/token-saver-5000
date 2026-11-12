"""Stdlib-only public-unicast address policy for outbound fetch adapters."""

import ipaddress


def normalize_address(
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Collapse IPv4-mapped IPv6 before applying the IPv4 policy."""
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def rejection_reason(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Return a denial reason, or None for a globally routable unicast address."""
    addr = normalize_address(addr)
    if addr.is_loopback:
        return "loopback address not allowed"
    if addr.is_link_local:
        return "link-local address not allowed"
    # ipaddress considers deprecated site-local space global.
    if isinstance(addr, ipaddress.IPv6Address) and addr.is_site_local:
        return "site-local address not allowed"
    if addr.is_multicast:
        return "multicast address not allowed"
    if addr.is_unspecified:
        return "unspecified address not allowed"
    if addr.is_private:
        return "private address not allowed"
    if addr.is_reserved:
        return f"reserved IPv{addr.version} address not allowed"
    if not addr.is_global:
        return "non-public address not allowed"
    return None
