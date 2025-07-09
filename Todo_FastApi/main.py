from fastapi import FastAPI
import json
from pydantic import BaseModel


app = FastAPI()
todos = []


class Todo(BaseModel):
    id: int
    task: str


@app.get("/get_todo")
async def getTodos():
    if len(todos) <= 0:
        return json.dumps({"status": "success", "message": "No todos exists"}, indent=4)
    return [t.dict() for t in todos]


@app.post("/create")
async def createTodo(todo: Todo):
    todos.append(todo)
    return {"status": "success", "message": "Todo created", "todos": todos}


@app.delete("/delete/{id}")
async def deleteTodo(id: int):
    i = 0
    for todo in todos:
        if todo.id == id:
            todos.pop(i)
            return {"status": "success", "message": "Todo deleted", "todos": todos}
        i = i + 1
        return {"message": "unable to delete"}


@app.put("/update")
async def update(t: Todo):
    for todo in todos:
        if todo.id == t.id:
            todo.task = t.task
            return {
                "status": "success",
                "message": "todo updated successfully",
                "todo": todo,
            }
    return {"status": "success", "message": "unable to update todo"}
