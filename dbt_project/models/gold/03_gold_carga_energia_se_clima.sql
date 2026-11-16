{{
    config(
        alias="gold_carga_energia_se_clima",
        materialized="table"
    )
}}

with carga as (
    select *
    from {{ ref('03_gold_carga_energia_se') }}
),

clima as (
    select *
    from {{ ref('02_silver_temperatura_se') }}
)

select
    c.sk_carga_diaria,
    c.data_referencia,
    c.id_subsistema,
    c.nom_subsistema,
    c.carga_energia_mwmed,
    c.regime_medicao_ons,
    w.temp_media_se,
    w.temp_max_se,
    w.temp_min_se,
    w.temp_aparente_se,
    w.temp_max_sp
from carga c
inner join clima w
    on c.data_referencia = w.data_referencia
order by c.data_referencia asc
