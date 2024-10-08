"""Add ProcessedVideo model

Revision ID: 9bd10ae46137
Revises: e316b233b4e4
Create Date: 2024-06-03 15:47:47.864911

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9bd10ae46137'
down_revision = 'e316b233b4e4'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('video', schema=None) as batch_op:
        batch_op.drop_column('is_processed')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('video', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_processed', sa.BOOLEAN(), nullable=True))

    # ### end Alembic commands ###
