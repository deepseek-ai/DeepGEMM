import pytest
import torch

from deep_gemm.mega import _pack_fp4_weight_tiles


def test_tile_packets_require_complete_cta_pairs():
    weight = torch.empty((1, 128, 128), dtype=torch.int8)
    sf = torch.empty((1, 128, 2), dtype=torch.int32)

    with pytest.raises(
            AssertionError,
            match='require an even number of N tiles'):
        _pack_fp4_weight_tiles(weight, sf)


def test_tile_packets_keep_cluster_peers_and_scales_adjacent():
    num_groups = 2
    n = 512
    logical_k = 512
    packed_k = logical_k // 2
    packed_sf_k = logical_k // 128

    weight = torch.arange(
        num_groups * n * packed_k, dtype=torch.int64
    ).to(torch.int8).reshape(num_groups, n, packed_k)
    sf = torch.arange(
        num_groups * n * packed_sf_k, dtype=torch.int32
    ).reshape(num_groups, n, packed_sf_k)

    packed_weight, packed_sf = _pack_fp4_weight_tiles(weight, sf)

    assert packed_weight.untyped_storage().data_ptr() == \
        packed_sf.untyped_storage().data_ptr()
    assert packed_sf.data_ptr() - packed_weight.data_ptr() == 128 * 128
    assert packed_weight.stride() == (17408, 128, 1)
    assert packed_sf.stride() == (4352, 128, 1)

    num_n_clusters = n // 128 // 2
    num_k_tiles = logical_k // 256
    for group_idx in range(num_groups):
        for cluster_idx in range(num_n_clusters):
            for k_tile_idx in range(num_k_tiles):
                for peer_idx in range(2):
                    packet_idx = (
                        ((group_idx * num_n_clusters + cluster_idx) *
                         num_k_tiles + k_tile_idx) * 2 + peer_idx
                    )
                    n_start = (cluster_idx * 2 + peer_idx) * 128
                    k_start = k_tile_idx * 128
                    expected_weight = weight[
                        group_idx,
                        n_start:n_start + 128,
                        k_start:k_start + 128,
                    ]
                    expected_sf = sf[
                        group_idx,
                        n_start:n_start + 128,
                        k_tile_idx * 2:k_tile_idx * 2 + 2,
                    ].T
                    assert torch.equal(
                        packed_weight[packet_idx], expected_weight)
                    assert torch.equal(packed_sf[packet_idx], expected_sf)


def test_k128_compute_tiles_are_the_two_halves_of_one_packet():
    n = 256
    logical_k = 256
    weight = torch.arange(
        n * logical_k // 2, dtype=torch.int64
    ).to(torch.int8).reshape(1, n, logical_k // 2)
    sf = torch.arange(
        n * logical_k // 128, dtype=torch.int32
    ).reshape(1, n, logical_k // 128)

    packed_weight, packed_sf = _pack_fp4_weight_tiles(weight, sf)

    for peer_idx in range(2):
        packet_idx = peer_idx
        n_start = peer_idx * 128
        for packet_half in range(2):
            packed_k_start = packet_half * 64
            expected_weight = weight[
                0,
                n_start:n_start + 128,
                packed_k_start:packed_k_start + 64,
            ]
            expected_sf = sf[
                0,
                n_start:n_start + 128,
                packet_half:packet_half + 1,
            ].T
            assert torch.equal(
                packed_weight[
                    packet_idx, :, packed_k_start:packed_k_start + 64],
                expected_weight)
            assert torch.equal(
                packed_sf[packet_idx, packet_half:packet_half + 1, :],
                expected_sf)
