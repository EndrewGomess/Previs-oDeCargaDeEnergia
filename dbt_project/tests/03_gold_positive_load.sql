-- Garante que não existem valores de carga menores ou iguais a zero na camada Gold SE
select
    data_referencia,
    carga_energia_mwmed
from {{ ref('03_gold_carga_energia_se') }}
where carga_energia_mwmed <= 0
