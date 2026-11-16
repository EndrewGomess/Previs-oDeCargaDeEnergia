{{
    config(
        alias="silver_carga_energia",
        materialized="table"
    )
}}

with bronze_raw as (
    select
        id_subsistema,
        nom_subsistema,
        din_instante,
        val_cargaenergiamwmed
    from read_parquet('../data/bronze/CARGA_ENERGIA_*.parquet', union_by_name=true)
),

typed_and_cleaned as (
    select
        md5(
            upper(trim(cast(id_subsistema as varchar)))
            || '-'
            || cast(cast(din_instante as date) as varchar)
        ) as sk_carga_diaria,
        upper(trim(cast(id_subsistema as varchar))) as id_subsistema,
        trim(cast(nom_subsistema as varchar)) as nom_subsistema,
        cast(din_instante as date) as data_referencia,
        try_cast(val_cargaenergiamwmed as double) as carga_energia_mwmed,
        cast(din_instante as timestamp) as din_instante_raw
    from bronze_raw
    where
        id_subsistema is not null
        and din_instante is not null
        and val_cargaenergiamwmed is not null
)

select
    sk_carga_diaria,
    id_subsistema,
    nom_subsistema,
    data_referencia,
    carga_energia_mwmed
from typed_and_cleaned
qualify row_number() over (
    partition by id_subsistema, data_referencia
    order by din_instante_raw desc
) = 1
