{{
    config(
        alias="silver_temperatura_se",
        materialized="table"
    )
}}

with bronze_weather as (
    select
        cast(data_referencia as date) as data_referencia,
        trim(cast(cidade as varchar)) as cidade,
        upper(trim(cast(uf as varchar))) as uf,
        cast(peso_carga as double) as peso_carga,
        cast(temp_media_c as double) as temp_media_c,
        cast(temp_max_c as double) as temp_max_c,
        cast(temp_min_c as double) as temp_min_c,
        cast(temp_aparente_c as double) as temp_aparente_c
    from read_parquet('../data/bronze/TEMPERATURA_SE_2018_2026.parquet')
)

select
    data_referencia,
    sum(temp_media_c * peso_carga) / sum(peso_carga) as temp_media_se,
    sum(temp_max_c * peso_carga) / sum(peso_carga) as temp_max_se,
    sum(temp_min_c * peso_carga) / sum(peso_carga) as temp_min_se,
    sum(temp_aparente_c * peso_carga) / sum(peso_carga) as temp_aparente_se,
    max(case when uf = 'SP' then temp_max_c end) as temp_max_sp
from bronze_weather
group by data_referencia
order by data_referencia asc
