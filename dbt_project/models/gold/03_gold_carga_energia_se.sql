{{
    config(
        alias="gold_carga_energia_se",
        materialized="table"
    )
}}

with silver_se as (
    select
        sk_carga_diaria,
        id_subsistema,
        nom_subsistema,
        data_referencia,
        carga_energia_mwmed
    from {{ ref('02_silver_carga_energia') }}
    where id_subsistema = 'SE'
)

select
    sk_carga_diaria,
    data_referencia,
    id_subsistema,
    nom_subsistema,
    carga_energia_mwmed,
    case
        when data_referencia < date '2021-03-01' then '1_despachada'
        when data_referencia < date '2023-04-29' then '2_nao_despachada'
        else '3_com_mmgd'
    end as regime_medicao_ons
from silver_se
order by data_referencia asc
