import db

from sqlalchemy import Column, Integer, String, JSON, LargeBinary
from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import ForeignKey


class Experiment(db.Base):
    __tablename__ = 'experiment'

    id = Column(Integer, primary_key=True)
    task_type = Column(String, nullable=False)
    task_parameters = Column(JSON)
    execution = relationship("Execution")

    def __init__(self, task_type, task_parameters):
        self.task_type = task_type
        self.task_parameters = task_parameters
    
    def __repr__(self) -> str:
        return f'Experiment({self.id}, {self.type}'
    
    def __str__(self) -> str:
        return f'{self.id} {self.task_type}'

class Execution(db.Base):
    __tablename__ = 'execution'

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiment.id'))
    execution_model = Column(String, nullable=False)
    executionBytes = Column(LargeBinary, nullable=False)
    parameters = Column(JSON)
    train_results = Column(JSON)
    test_results = Column(JSON)

    def __init__(self, experiment_id, execution_model, executionBytes, parameters, train_results, test_results):
        self.experiment_id = experiment_id
        self.execution_model = execution_model
        self.executionBytes = executionBytes
        self.parameters = parameters
        self.train_results = train_results
        self.test_results = test_results
    
    def __repr__(self) -> str:
        return f'Execution({self.id}, {self.experiment_id}, {self.model}'
    
    def __str__(self) -> str:
        return f'{self.id} {self.experiment_id} {self.model}'