"""updated db

Revision ID: 56f488da78b2
Revises: 
Create Date: 2024-11-13 22:42:31.336794

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '56f488da78b2'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('prediction_data', schema=None) as batch_op:
        batch_op.add_column(sa.Column('wealth_quantile', sa.String(length=20), nullable=False))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('prediction_data', schema=None) as batch_op:
        batch_op.drop_column('wealth_quantile')

    # ### end Alembic commands ###
